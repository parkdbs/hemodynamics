function [x_next, MAP, CO, SVR] = hemodynamic_plant(x, u_f, u_v, dt)
    % HEMODYNAMIC_PLANT Simulates one discrete time step of the cardiovascular model.
    % 
    % INPUT VARIABLES:
    %   x   - Current state vector [Vc; Vp; Cp; Ce; B] 
    %   u_f - Intravenous fluid infusion rate (mL/min)
    %   u_v - Vasopressor infusion rate (mcg/min)
    %   dt  - Time step size for Euler integration (minutes)
    
    % =========================================================================
    % PLANT PARAMETERS (Patient-Specific Physiological Constants)
    % =========================================================================
    
    % --- Baseline Hemodynamics ---
    % These represent the patient's resting state before any intervention.
    p.CO_base  = 4.5;   % Baseline Cardiac Output (L/min): The heart's natural resting blood flow.
    p.SVR_base = 13.3;  % Baseline Systemic Vascular Resistance (mmHg / (L/min)): Resting blood vessel tone.
    p.MAP_base = p.CO_base * p.SVR_base; % Baseline Mean Arterial Pressure (mmHg): Expected ~60 mmHg.
    
    % --- Fluid Volume Kinetics (2-Compartment Model) ---
    % These dictate how intravenous fluid moves through the body.
    p.k12      = 0.08;  % Transfer central to peripheral (min^-1): Rate at which fluid leaks from blood vessels into tissues (capillary leakage).
    p.k21      = 0.02;  % Transfer peripheral to central (min^-1): Rate at which fluid returns from tissues to vessels (lymphatic/venous return).
    p.k_ur     = 0.005; % Elimination rate (min^-1): Rate at which fluid is permanently lost via urine and insensible losses.
    
    % --- Frank-Starling Mechanism (Fluid -> Cardiac Output) ---
    % These define the non-linear curve of how preload (fluid) increases heart output.
    p.dCO_max  = 2.5;   % Max CO increase (L/min): The absolute ceiling of the heart's pumping capacity due to extra volume.
    p.V_50     = 400;   % Half-maximal volume (mL): The amount of central fluid (Vc) needed to reach 50% of the max cardiac output increase.
    
    % --- Vasopressor Pharmacokinetics (PK) ---
    % These define how the body distributes and clears the drug.
    p.V_d      = 5.0;   % Volume of distribution (L): The apparent anatomical volume the drug dilutes into.
    p.k_elim   = 0.25;  % Elimination rate (min^-1): How fast the liver and kidneys clear the drug from the plasma.
    p.k_e0     = 0.3;   % Effect-site transfer rate (min^-1): The delay representing how fast drug moves from plasma to the vessel wall receptors.
    
    % --- Vasopressor Pharmacodynamics (PD) ---
    % These define the non-linear curve of how the drug actually constricts vessels.
    p.E_max    = 10.0;  % Maximum SVR increase: The absolute physical limit of how much the drug can constrict the blood vessels.
    p.C_50     = 5.0;   % Half-maximal concentration (mcg/L): The effect-site concentration (Ce) needed to reach exactly 50% of the E_max.
    
    % --- Autonomic Nervous System (Baroreflex) ---
    % These define the body's natural negative feedback loop.
    p.tau_b    = 2.0;   % Baroreflex time constant (min): How quickly the nervous system responds to sudden pressure changes.
    p.K_b      = 0.05;  % Baroreflex gain: The strength/aggressiveness of the body's attempt to counteract the drug and return to the baseline MAP.
    
    % =========================================================================
    % EXTRACT STATES
    % =========================================================================
    Vc = x(1); % Central fluid volume change (mL)
    Vp = x(2); % Peripheral fluid volume change (mL)
    Cp = x(3); % Plasma concentration of vasopressor (mcg/L)
    Ce = x(4); % Effect-site concentration of vasopressor (mcg/L)
    B  = x(5); % Baroreflex sympathetic tone effect
    
    % =========================================================================
    % 1. ALGEBRAIC EQUATIONS (Hemodynamics)
    % =========================================================================
    % Cardiac Output (Frank-Starling saturation)
    CO = p.CO_base + (p.dCO_max * max(0, Vc)) / (p.V_50 + max(0, Vc));
    
    % Systemic Vascular Resistance (E_max saturation + Baroreflex)
    SVR = p.SVR_base + (p.E_max * Ce) / (p.C_50 + Ce) + B;
    
    % Mean Arterial Pressure
    MAP = CO * SVR;
    
    % =========================================================================
    % 2. DIFFERENTIAL EQUATIONS (Rates of Change)
    % =========================================================================
    % Fluid Kinetics
    dVc = u_f - p.k12*Vc + p.k21*Vp - p.k_ur*Vc;
    dVp = p.k12*Vc - p.k21*Vp;
    
    % Vasopressor PK/PD
    dCp = (u_v / p.V_d) - p.k_elim*Cp;
    dCe = p.k_e0 * (Cp - Ce);
    
    % Baroreflex (Autonomic feedback)
    dB  = (1 / p.tau_b) * (p.K_b * (p.MAP_base - MAP) - B);
    
    % =========================================================================
    % 3. DISCRETE STATE UPDATE
    % =========================================================================
    x_next = x + dt * [dVc; dVp; dCp; dCe; dB];
end