"""
Experimental Scenario Manager for MADRL Experiments

Implements the current 3-axis thesis setup:
- E1: Energy flexibility
- E2: Carbon emissions reduction
- E3: Economic cost optimization with integrated flexibility and carbon context
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for experimental scenario"""
    name: str  # 'E1', 'E2', 'E3'
    description: str
    
    # Tariff settings
    use_time_of_use: bool
    use_real_time_pricing: bool
    tariff_multiplier: float  # Base rate multiplier
    
    # Outage settings
    enable_outages: bool
    outage_frequency: float  # Outages per year
    outage_duration_min: int  # Hours
    outage_duration_max: int  # Hours
    
    # Legacy signal settings retained for backward compatibility with older runs.
    enable_dr_signals: bool
    dr_signal_threshold: float
    dr_response_required: bool
    
    # Reward weights (flexibility, carbon, cost)
    reward_weights: Dict[str, float]
    
    # KPI focus
    primary_kpi: str  # Which KPI to optimize for


class ScenarioManager:
    """Manages experimental scenarios"""
    
    # E1: Flexibility Focus
    E1_CONFIG = ScenarioConfig(
        name="E1",
        description="Flexibility Scenario - RTP tariff, no outages, normal operation",
        use_time_of_use=False,
        use_real_time_pricing=True,
        tariff_multiplier=1.0,
        enable_outages=False,
        outage_frequency=0,
        outage_duration_min=2,
        outage_duration_max=8,
        enable_dr_signals=False,
        dr_signal_threshold=0.8,
        dr_response_required=False,
        reward_weights={"flex": 0.6, "carbon": 0.2, "cost": 0.2},
        primary_kpi="peak_average"
    )
    
    # E2: Carbon Emissions Focus
    E2_CONFIG = ScenarioConfig(
        name="E2",
        description="Carbon Emissions Scenario - carbon-aware operation with scheduled/random outages",
        use_time_of_use=False,
        use_real_time_pricing=False,
        tariff_multiplier=1.0,
        enable_outages=True,
        outage_frequency=52,  # ~1 per week
        outage_duration_min=2,
        outage_duration_max=8,
        enable_dr_signals=False,
        dr_signal_threshold=0.8,
        dr_response_required=False,
        reward_weights={"flex": 0.2, "carbon": 0.6, "cost": 0.2},
        primary_kpi="carbon_emissions_total"
    )
    
    # E3: Economic Cost Focus
    E3_CONFIG = ScenarioConfig(
        name="E3",
        description="Cost Scenario - RTP pricing with integrated flexibility and carbon objectives",
        use_time_of_use=False,
        use_real_time_pricing=True,
        tariff_multiplier=1.2,
        enable_outages=True,
        outage_frequency=52,
        outage_duration_min=2,
        outage_duration_max=8,
        enable_dr_signals=False,
        dr_signal_threshold=0.75,
        dr_response_required=False,
        reward_weights={"flex": 1/3, "carbon": 1/3, "cost": 1/3},
        primary_kpi="electricity_cost_total"
    )
    
    SCENARIOS = {
        "E1": E1_CONFIG,
        "E2": E2_CONFIG,
        "E3": E3_CONFIG,
    }
    
    def __init__(self):
        """Initialize scenario manager"""
        self.current_scenario = None
        self.current_config = None
        self.outage_schedule = []
    
    def select_scenario(self, scenario_name: str) -> ScenarioConfig:
        """Select and initialize a scenario
        
        Args:
            scenario_name: 'E1', 'E2', or 'E3'
        
        Returns:
            ScenarioConfig for the selected scenario
        """
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Must be one of {list(self.SCENARIOS.keys())}")
        
        self.current_scenario = scenario_name
        self.current_config = self.SCENARIOS[scenario_name]
        
        # Generate outage schedule if needed
        if self.current_config.enable_outages:
            self.outage_schedule = self._generate_outage_schedule()
        
        logger.info(f"Selected scenario {scenario_name}: {self.current_config.description}")
        
        return self.current_config
    
    def _generate_outage_schedule(self, year_hours: int = 8760, seed: int = None) -> List[Tuple[int, int]]:
        """Generate random outage schedule for the year
        
        Args:
            year_hours: Total hours in year (8760 for standard year)
            seed: Random seed for reproducibility
        
        Returns:
            List of (start_hour, end_hour) tuples for outages
        """
        if seed is not None:
            np.random.seed(seed)
        
        config = self.current_config
        n_outages = np.random.poisson(config.outage_frequency)  # Random number of outages
        
        outages = []
        for _ in range(n_outages):
            # Random start hour
            start_hour = np.random.randint(0, year_hours)
            # Random duration
            duration = np.random.randint(config.outage_duration_min, config.outage_duration_max + 1)
            end_hour = min(start_hour + duration, year_hours)
            
            outages.append((start_hour, end_hour))
        
        # Sort by start hour
        outages.sort(key=lambda x: x[0])
        
        logger.info(f"Generated {len(outages)} outages for scenario {self.current_scenario}")
        
        return outages
    
    def is_outage_hour(self, hour: int) -> bool:
        """Check if given hour is during an outage
        
        Args:
            hour: Hour of the year (0-8759)
        
        Returns:
            True if hour is during an outage, False otherwise
        """
        if not self.current_config.enable_outages:
            return False
        
        for start, end in self.outage_schedule:
            if start <= hour < end:
                return True
        
        return False
    
    def get_tariff_multiplier(self, hour: int) -> float:
        """Get tariff multiplier for given hour (TOU or RTP)
        
        Args:
            hour: Hour of the day (0-23)
        
        Returns:
            Tariff multiplier (1.0 = base rate)
        """
        if self.current_config.use_time_of_use:
            # TOU: Peak (8-22h) = 1.5x, Off-peak = 0.5x
            hour_of_day = hour % 24
            if 8 <= hour_of_day < 22:
                return 1.5 * self.current_config.tariff_multiplier
            else:
                return 0.5 * self.current_config.tariff_multiplier
        
        elif self.current_config.use_real_time_pricing:
            # RTP: Simulate variable pricing with sinusoidal pattern + noise
            hour_of_year = hour % 8760
            base_price = 1.0 + 0.5 * np.sin(2 * np.pi * hour_of_year / 8760)
            noise = np.random.normal(0, 0.1)
            rtp = np.clip(base_price + noise, 0.5, 1.5)
            return rtp * self.current_config.tariff_multiplier
        
        else:
            return self.current_config.tariff_multiplier
    
    def get_dr_signal(self, hour: int, grid_load: float, threshold: float = 0.8) -> Optional[Dict]:
        """Get demand response signal if scenario enables it
        
        Args:
            hour: Current hour
            grid_load: Current grid load (0-1 normalized)
            threshold: Load threshold for triggering DR signal
        
        Returns:
            DR signal dict or None if not applicable
        """
        if not self.current_config.enable_dr_signals:
            return None
        
        # Trigger DR signal if grid load is high
        if grid_load > self.current_config.dr_signal_threshold:
            return {
                "type": "load_reduction",
                "target_reduction": 0.2 + 0.3 * (grid_load - threshold),  # 20-50% reduction
                "duration_hours": 2,
                "reward_bonus": 0.5 if self.current_config.dr_response_required else 0.0,
                "penalty": -0.3 if self.current_config.dr_response_required else 0.0
            }
        
        return None
    
    def apply_scenario_modifications(self, env) -> None:
        """Apply scenario-specific modifications to CityLearn environment
        
        Args:
            env: CityLearnEnv instance
        """
        if self.current_config is None:
            logger.warning("No scenario selected. Call select_scenario() first.")
            return
        
        logger.info(f"Applying modifications for scenario {self.current_scenario}")
        
        # Apply tariff modifications
        if self.current_config.use_time_of_use:
            logger.info("Enabling Time-of-Use (TOU) tariff for this scenario")
        elif self.current_config.use_real_time_pricing:
            logger.info("Enabling Real-Time Pricing (RTP) for this scenario")
        
        # Apply outage settings
        if self.current_config.enable_outages:
            logger.info(f"Outages enabled: {len(self.outage_schedule)} scheduled outages")
        
        # Legacy DR settings are intentionally disabled for the current
        # flexibility-carbon-cost thesis axes.
        if self.current_config.enable_dr_signals:
            logger.info("Legacy demand-response signals enabled")
    
    def get_scenario_description(self) -> Dict:
        """Get human-readable scenario description"""
        if self.current_config is None:
            return {}
        
        return {
            "scenario": self.current_scenario,
            "description": self.current_config.description,
            "tariff": "TOU" if self.current_config.use_time_of_use else ("RTP" if self.current_config.use_real_time_pricing else "FLAT"),
            "outages_enabled": self.current_config.enable_outages,
            "n_outages": len(self.outage_schedule) if self.current_config.enable_outages else 0,
            "legacy_dr_signals_enabled": self.current_config.enable_dr_signals,
            "reward_weights": self.current_config.reward_weights,
            "primary_kpi": self.current_config.primary_kpi
        }
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, str]:
        """Get all available scenarios"""
        return {
            "E1": ScenarioManager.E1_CONFIG.description,
            "E2": ScenarioManager.E2_CONFIG.description,
            "E3": ScenarioManager.E3_CONFIG.description,
        }


class ExperimentalDesign:
    """Manages experimental design per thesis specifications
    
    Design:
    - 4 algorithms: HAPPO, MASAC, MATD3, MAAC
    - 3 scenarios: E1 (flexibility), E2 (carbon), E3 (cost)
    - 10 random seeds
    - Total: 4 × 3 × 10 = 120 experiments
    """
    
    N_ALGORITHMS = 4
    N_SCENARIOS = 3
    N_SEEDS = 10
    N_EXPERIMENTS = N_ALGORITHMS * N_SCENARIOS * N_SEEDS
    
    ALGORITHMS = ["HAPPO", "MASAC", "MATD3", "MAAC"]
    SCENARIOS = ["E1", "E2", "E3"]
    SEEDS = list(range(42, 42 + N_SEEDS))
    
    def __init__(self):
        """Initialize experiment design"""
        self.experiments = self._generate_experiment_plan()
    
    def _generate_experiment_plan(self) -> List[Dict]:
        """Generate complete experiment plan"""
        experiments = []
        exp_id = 1
        
        for algo in self.ALGORITHMS:
            for scenario in self.SCENARIOS:
                for seed in self.SEEDS:
                    experiments.append({
                        "id": exp_id,
                        "algorithm": algo,
                        "scenario": scenario,
                        "seed": seed,
                        "n_episodes": 10,
                        "n_timesteps": 8760,  # 1 year hourly
                    })
                    exp_id += 1
        
        return experiments
    
    def get_experiment(self, exp_id: int) -> Dict:
        """Get experiment configuration by ID"""
        if 1 <= exp_id <= len(self.experiments):
            return self.experiments[exp_id - 1]
        else:
            raise ValueError(f"Invalid experiment ID: {exp_id}")
    
    def get_experiments_for_algorithm(self, algorithm: str) -> List[Dict]:
        """Get all experiments for a specific algorithm"""
        return [e for e in self.experiments if e["algorithm"] == algorithm]
    
    def get_experiments_for_scenario(self, scenario: str) -> List[Dict]:
        """Get all experiments for a specific scenario"""
        return [e for e in self.experiments if e["scenario"] == scenario]
    
    def get_experiments_for_seed(self, seed: int) -> List[Dict]:
        """Get all experiments for a specific seed"""
        return [e for e in self.experiments if e["seed"] == seed]
    
    def get_total_experiments(self) -> int:
        """Get total number of experiments"""
        return len(self.experiments)
    
    def summary(self) -> Dict:
        """Get experimental design summary"""
        return {
            "total_experiments": self.N_EXPERIMENTS,
            "algorithms": self.ALGORITHMS,
            "n_algorithms": self.N_ALGORITHMS,
            "scenarios": self.SCENARIOS,
            "n_scenarios": self.N_SCENARIOS,
            "seeds": self.SEEDS,
            "n_seeds": self.N_SEEDS,
            "episodes_per_experiment": 10,
            "timesteps_per_episode": 8760,
            "total_timesteps": self.N_EXPERIMENTS * 10 * 8760
        }
