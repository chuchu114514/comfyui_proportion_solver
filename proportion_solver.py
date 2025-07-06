import numpy as np
import torch

# ==========================================================================================
# == 版本 1: 简单比例优化器 (Simple Proportion Solver)
# ==========================================================================================
class ProportionSolver:
    """
    一个执行梯度下降优化的节点，用于找到满足特定比例的三个参数 x, y, z。
    版本2：增加了P1, P2, P3作为可调输入。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_p1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_p2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_p3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "iterations": ("INT", {"default": 10000, "min": 100, "max": 100000, "step": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("x_final", "y_final", "z_final", "P1", "P2", "P3", "P4")
    FUNCTION = "solve"
    CATEGORY = "Utils/Logic"

    def solve(self, learning_rate, iterations, seed, target_p1, target_p2, target_p3):
        p_sum = target_p1 + target_p2 + target_p3
        if p_sum > 1.0:
            print(f"[ProportionSolver] Warning: Sum of target P1, P2, P3 ({p_sum:.2f}) is > 1. P4 will be 0.")
        target_p4 = max(0.0, 1.0 - p_sum)
        target_proportions = np.array([target_p1, target_p2, target_p3, target_p4])
        print(f"[ProportionSolver] Using Target Proportions: {np.round(target_proportions, 4)}")

        def calculate_proportions(x, y, z):
            if not (0 < x < 1 and 0 < y < 1 and 0 < z < 1): return np.array([0.0] * 4)
            y_adjusted = 1.1 * y
            P1 = 1 - x
            P2 = x * (1 - y_adjusted) + 0.05 * P1
            P3 = x * y_adjusted * (1 - z) + 0.05 * P2
            P4 = x * y_adjusted * z + 0.05 * P3
            return np.array([P1, P2, P3, P4])

        def loss_function(x, y, z):
            current_proportions = calculate_proportions(x, y, z)
            return np.sum((current_proportions - target_proportions) ** 2)

        def numerical_gradient(func, x_val, y_val, z_val, h=1e-6):
            grad_x = (func(x_val + h, y_val, z_val) - func(x_val - h, y_val, z_val)) / (2 * h)
            grad_y = (func(x_val, y_val + h, z_val) - func(x_val, y_val - h, z_val)) / (2 * h)
            grad_z = (func(x_val, y_val, z_val + h) - func(x_val, y_val, z_val - h)) / (2 * h)
            return grad_x, grad_y, grad_z
        
        rng = np.random.RandomState(seed)
        x, y, z = rng.uniform(0.1, 0.9, 3)
        for i in range(iterations):
            grad_x, grad_y, grad_z = numerical_gradient(loss_function, x, y, z)
            x -= learning_rate * grad_x
            y -= learning_rate * grad_y
            z -= learning_rate * grad_z
            x, y, z = np.clip([x, y, z], 0.001, 0.999)
        final_proportions = calculate_proportions(x, y, z)
        return (float(x), float(y), float(z), *final_proportions.astype(float))


# ==========================================================================================
# == 版本 2: 高级比例优化器 (Advanced Proportion Solver)
# ==========================================================================================
class ProportionSolverAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "target_p1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_p2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_p3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "hyperparameters": ("GROUP", {
                    "regularization_strength": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.1, "step": 0.0001}),
                    "stabilization_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "integer_strength": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.1, "step": 0.0001}),
                    "hundredths_strength": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.1, "step": 0.0001}),
                }),
                "annealing_restart": ("GROUP", {
                    "n_restarts": ("INT", {"default": 4, "min": 0, "max": 20}),
                    "initial_period": ("INT", {"default": 2000, "min": 100, "max": 10000, "step": 100}),
                    "period_multiplier": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "lr_max": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.1, "step": 0.001}),
                    "lr_min": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 0.001, "step": 1e-6, "display": "number"}),
                }),
                "adam_optimizer": ("GROUP", {
                    "beta1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.999, "step": 0.01}),
                    "beta2": ("FLOAT", {"default": 0.999, "min": 0.0, "max": 0.9999, "step": 0.001}),
                    "epsilon": ("FLOAT", {"default": 1e-8, "min": 1e-9, "max": 1e-6, "step": 1e-9, "display": "number"}),
                })
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", # Raw params
                    "FLOAT", "FLOAT", "FLOAT", "FLOAT", # Proportions
                    "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT") # Rounded params
    RETURN_NAMES = ("x", "y", "z", "a", "b", "c", "d",
                    "P1", "P2", "P3", "P4",
                    "x_rnd", "y_rnd", "z_rnd", "a_rnd", "b_rnd", "c_rnd", "d_rnd")
    FUNCTION = "solve_advanced"
    CATEGORY = "Utils/Logic"

    def solve_advanced(self, seed, target_p1, target_p2, target_p3, **kwargs):
        # Unpack kwargs from groups
        hp = kwargs.get('hyperparameters', {})
        anneal = kwargs.get('annealing_restart', {})
        adam = kwargs.get('adam_optimizer', {})

        # --- Assign all parameters from inputs ---
        p_sum = target_p1 + target_p2 + target_p3
        if p_sum > 1.0: print(f"[SolverAdvanced] Warning: Sum of targets > 1. P4 will be 0.")
        target_p4 = max(0.0, 1.0 - p_sum)
        TARGET_PROPORTIONS = np.array([target_p1, target_p2, target_p3, target_p4])

        regularization_strength = hp.get('regularization_strength', 0.0005)
        stabilization_weight = hp.get('stabilization_weight', 0.1)
        integer_strength = hp.get('integer_strength', 0.0005)
        hundredths_strength = hp.get('hundredths_strength', 0.0005)

        n_restarts = anneal.get('n_restarts', 4)
        initial_period = anneal.get('initial_period', 2000)
        period_multiplier = anneal.get('period_multiplier', 1.5)
        lr_max = anneal.get('lr_max', 0.008)
        lr_min = anneal.get('lr_min', 1e-6)

        beta1 = adam.get('beta1', 0.9)
        beta2 = adam.get('beta2', 0.999)
        epsilon = adam.get('epsilon', 1e-8)

        # --- Core Functions from Script ---
        def calculate_proportions(params):
            x, y, z, a, b, c, d = params
            if not (0 < x < 1 and 0 < y < 1 and 0 < z < 1): return np.array([1e6]*4)
            y_mod = 1.1 * y
            p1_base = 1 - x
            p2_base = x * (1 - y_mod) + 0.05 * p1_base
            p3_base = x * y_mod * (1 - z) + 0.05 * p2_base
            p4_base = x * y_mod * z + 0.05 * p3_base
            weighted_values = np.array([p1_base * a, p2_base * b, p3_base * c, p4_base * d])
            total = np.sum(weighted_values)
            return weighted_values / total if total > 1e-9 else np.array([0.0]*4)

        def loss_function(params):
            x, y, z, a, b, c, d = params
            mse_loss = np.sum((calculate_proportions(params) - TARGET_PROPORTIONS)**2)
            stab_loss = stabilization_weight * (np.maximum(0, 2.0 - a)**2 + (b - 8.0)**2 + np.maximum(0, 2.0 - c)**2 + (d - 4.0)**2)
            reg_loss = regularization_strength * (a + x*b + y*1.44*c + z*1.44*d)
            int_prox = lambda p: 0.5 * (1 - np.cos(2 * np.pi * p))
            int_loss = integer_strength * (int_prox(a) + int_prox(b) + int_prox(c) + int_prox(d))
            hund_loss = hundredths_strength * (int_prox(x*100) + int_prox(y*100) + int_prox(z*100))
            return mse_loss + stab_loss + reg_loss + int_loss + hund_loss

        def numerical_gradient(func, params, h=1e-6):
            grads = np.zeros_like(params)
            for i in range(len(params)):
                old_val = params[i]
                params[i] = old_val + h; fxh1 = func(params)
                params[i] = old_val - h; fxh2 = func(params)
                grads[i] = (fxh1 - fxh2) / (2 * h)
                params[i] = old_val
            return grads
        
        # --- Optimization Process ---
        rng = np.random.RandomState(seed)
        params = np.array([
            rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
            rng.uniform(2.0, 4.0), rng.uniform(7.0, 9.0), rng.uniform(2.0, 4.0), rng.uniform(3.0, 5.0)
        ])
        
        best_loss = float('inf')
        best_params = params.copy()
        
        total_iterations = 0
        current_period = float(initial_period)

        print(f"[SolverAdvanced] Starting optimization. Target: {np.round(TARGET_PROPORTIONS, 4)}")

        for restart in range(n_restarts + 1):
            print(f"--- [Restart {restart + 1}/{n_restarts + 1}] Period: {int(current_period)} iters ---")
            m, v = np.zeros_like(params), np.zeros_like(params)
            
            for i in range(int(current_period)):
                t = total_iterations + 1
                progress_in_cycle = i / current_period
                current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress_in_cycle))
                
                grads = numerical_gradient(loss_function, params)
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * (grads**2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                params -= current_lr * m_hat / (np.sqrt(v_hat) + epsilon)
                
                params[0:3] = np.clip(params[0:3], 0.001, 0.999)
                params[3:7] = np.clip(params[3:7], 0.001, 11.999)
                
                current_loss = loss_function(params)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = params.copy()
                
                total_iterations += 1
            current_period *= period_multiplier

        final_proportions = calculate_proportions(best_params)
        
        print("\n" + "="*50)
        print("[SolverAdvanced] --- Optimization Finished ---")
        print(f"Best Loss Found: {best_loss:.6f}")
        print(f"Final Params: x={best_params[0]:.4f}, y={best_params[1]:.4f}, z={best_params[2]:.4f}, a={best_params[3]:.4f}, b={best_params[4]:.4f}, c={best_params[5]:.4f}, d={best_params[6]:.4f}")
        print("="*50)

        # --- Prepare outputs ---
        x_r, y_r, z_r = round(best_params[0], 2), round(best_params[1], 2), round(best_params[2], 2)
        a_r, b_r, c_r, d_r = round(best_params[3]), round(best_params[4]), round(best_params[5]), round(best_params[6])
        
        raw_params_out = tuple(p.astype(float) for p in best_params)
        proportions_out = tuple(p.astype(float) for p in final_proportions)
        rounded_params_out = (float(x_r), float(y_r), float(z_r), float(a_r), float(b_r), float(c_r), float(d_r))
        
        return (*raw_params_out, *proportions_out, *rounded_params_out)


# ==========================================================================================
# == Node Mappings for ComfyUI
# ==========================================================================================
NODE_CLASS_MAPPINGS = {
    "ProportionSolver": ProportionSolver,
    "ProportionSolverAdvanced": ProportionSolverAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProportionSolver": "Proportion Optimization Solver",
    "ProportionSolverAdvanced": "Proportion Optimization Solver (Advanced)",
}