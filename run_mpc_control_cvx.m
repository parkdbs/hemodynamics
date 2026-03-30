function run_mpc_control_cvx()
    % RUN_MPC_CONTROL_CVX Simulates closed-loop MAP control using CVX.
    % Uses Successive Linearization (LTV-MPC) to handle the non-linear plant.

    % --- 1. Simulation & Controller Setup ---
    T_end = 60;          % Simulate for 60 minutes
    dt = 1;              % 1 minute control interval
    MAP_target = 65;     % Target MAP (mmHg)
    
    % MPC Tuning Weights
    Q  = 10.0;           % Penalty for MAP tracking error
    R1 = 0.5;            % Penalty on fluid administration rate
    R2 = 5.0;            % Penalty on vasopressor rate of change
    Np = 5;              % Prediction horizon (minutes)
    
    % --- 3. Initialization ---
    x_current = [0; 0; 0; 0; 0]; % True patient states
    u_prev = [0; 0];             % Previous control moves
    
    % Preallocate arrays for plotting
    history.MAP = zeros(T_end, 1);
    history.u_f = zeros(T_end, 1);
    history.u_v = zeros(T_end, 1);
    history.t   = 1:T_end;

    disp('Running CVX-based LTV-MPC closed-loop simulation...');
    
    % --- 4. Main Closed-Loop ---
    for k = 1:T_end
        % A. Measure current output from the TRUE non-linear plant
        [~, MAP_current, ~, ~] = hemodynamic_plant(x_current, 0, 0, 0);
        history.MAP(k) = MAP_current;
        
        % B. Successive Linearization
        % Obtain linear state-space matrices around the current operating point
        [Ad, Bd, Cd, x_next0, y0] = get_linear_model(x_current, u_prev, dt);
        
        % Calculate affine constants so the linear model matches the non-linear 
        % model exactly at the current operating point.
        x_affine = x_next0 - Ad * x_current - Bd * u_prev(:);
        y_affine = y0 - Cd * x_current;
        
        % C. CVX Optimization Block
        cvx_begin quiet
            variables U(2, Np) X(5, Np+1) Y(1, Np)
            
            % Construct the cost function using CVX's 'square' function for DCP compliance
            J_cost = 0;
            for i = 1:Np
                J_cost = J_cost + Q * square(Y(i) - MAP_target);
                J_cost = J_cost + R1 * square(U(1,i));
                if i == 1
                    J_cost = J_cost + R2 * square(U(2,i) - u_prev(2));
                else
                    J_cost = J_cost + R2 * square(U(2,i) - U(2,i-1));
                end
            end
            
            minimize(J_cost)
            
            subject to
                X(:,1) == x_current; % Initial state constraint
                
                for i = 1:Np
                    % Linearized System Dynamics
                    X(:,i+1) == Ad * X(:,i) + Bd * U(:,i) + x_affine;
                    
                    % Linearized Output Equation
                    Y(i) == Cd * X(:,i) + y_affine;
                    
                    % Actuator Constraints (Fluid: 0-100 mL/min, Vaso: 0-20 mcg/min)
                    U(1,i) >= 0; 
                    U(1,i) <= 100;
                    U(2,i) >= 0; 
                    U(2,i) <= 20;
                end
        cvx_end
        
        % D. Extract optimal control move (first step of the horizon)
        % Fallback in case CVX fails to find a feasible solution
        if strcmp(cvx_status, 'Solved')
            u_opt = U(:,1);
        else
            disp(['CVX failed at step ', num2str(k), ' Status: ', cvx_status]);
            u_opt = u_prev; % Hold previous input
        end
        
        % Log control moves and update memory
        history.u_f(k) = u_opt(1);
        history.u_v(k) = u_opt(2);
        u_prev = u_opt;
        
        % E. Apply control to the TRUE NON-LINEAR PLANT to advance simulation time
        [x_current, ~, ~, ~] = hemodynamic_plant(x_current, u_opt(1), u_opt(2), dt);
    end
    
    % --- 5. Plotting ---
    plot_results(history, MAP_target);
end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [Ad, Bd, Cd, x_next0, y0] = get_linear_model(x0, u0, dt)
    % Numerically calculates the Jacobian matrices (A, B, C) for the non-linear plant.
    nx = 5; nu = 2;
    Ad = zeros(nx, nx); Bd = zeros(nx, nu); Cd = zeros(1, nx);
    
    % Small perturbations for numerical differentiation
    dx = 1e-5; du = 1e-5;
    
    % Base evaluation
    [x_next0, y0, ~, ~] = hemodynamic_plant(x0, u0(1), u0(2), dt);
    
    % Calculate A (State Jacobian) and C (Output Jacobian)
    for i = 1:nx
        x_pert = x0; 
        x_pert(i) = x_pert(i) + dx;
        [x_next_pert, y_pert, ~, ~] = hemodynamic_plant(x_pert, u0(1), u0(2), dt);
        
        Ad(:, i) = (x_next_pert - x_next0) / dx;
        Cd(1, i)  = (y_pert - y0) / dx;
    end
    
    % Calculate B (Input Jacobian)
    for i = 1:nu
        u_pert = u0; 
        u_pert(i) = u_pert(i) + du;
        [x_next_pert, ~, ~, ~] = hemodynamic_plant(x0, u_pert(1), u_pert(2), dt);
        
        Bd(:, i) = (x_next_pert - x_next0) / du;
    end
end

function plot_results(history, MAP_target)
    figure('Name', 'CVX LTV-MPC Closed-Loop', 'Color', 'w');
    
    subplot(3,1,1);
    stairs(history.t, history.u_f, 'b', 'LineWidth', 2);
    title('Fluid Resuscitation (u_{fluid})'); ylabel('mL/min'); grid on;
    
    subplot(3,1,2);
    stairs(history.t, history.u_v, 'r', 'LineWidth', 2);
    title('Vasopressor Infusion (u_{vaso})'); ylabel('mcg/min'); grid on;
    
    subplot(3,1,3);
    plot(history.t, history.MAP, 'y', 'LineWidth', 2);
    yline(MAP_target, 'r--', 'Target MAP', 'LineWidth', 1.5);
    title('Mean Arterial Pressure (MAP)'); 
    xlabel('Time (minutes)'); ylabel('mmHg'); ylim([55 75]); grid on;
end