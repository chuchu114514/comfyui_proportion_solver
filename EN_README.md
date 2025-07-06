Here's the English translation of the provided text:

# comfyui\\comfyui\_proportion\_solver

**Note:** This project was written by AI (gemini\_advanced-gemini\_2.5\_pro) to provide an efficient proportion optimization solution. Through multiple rounds of dialogue, the project iteratively evolved from core code to its final convenient form as ComfyUI nodes.

-----

## Environment Requirements

  * **Python Dependencies:** `numpy`
  * **Deep Learning Framework:** `torch` (ComfyUI users typically already have this environment, so no special instructions are provided.)

-----

## Usage Tutorial

1.  **Install Nodes:** Place the project files into ComfyUI's `custom_nodes` folder.
2.  **Launch ComfyUI:** Restart the ComfyUI interface.
3.  **Load Workflow:** We recommend using our provided workflow file for an optimal experience: image.png
4.  **Download Acceleration LoRA:** The example uses a hyper-LoRA, but you can certainly use other acceleration LoRAs (and control the proportions through node adjustments, where appropriate, without affecting quality).

-----

## Plugin Node Introduction

This plugin includes two core nodes, each designed for proportion optimization tasks of varying complexity:

### 1\. Proportion Optimization Solver

This is a **basic** tool for proportion optimization.

  * **Core Functionality:** Based on your three target "generation rates" (`target_generation_rate_1` to `3`), it automatically calculates three "denoise" parameters (`denoise_1` to `3`) that best match these ratios, using a **standard gradient descent algorithm**.
  * **Applicable Scenarios:** When you need a quick and direct solution to a basic proportion allocation problem, this node is an ideal starting choice.

-----

### 2\. Proportion Optimization Solver (Advanced)

This is the **expert version** of the solver, designed for more complex and refined optimization tasks.

  * **Core Functionality:**
      * It not only calculates three "denoise" parameters but also **additionally calculates four "step" parameters** (`step_1` to `4`) for more flexible control.
      * It employs a more powerful **Adam optimizer** combined with **Cosine Annealing Learning Rate** and **Warm Restarts** strategies, enabling it to explore and find high-quality solutions more effectively.
  * **Advanced Control:** You can precisely guide the optimization process through a multitude of **hyperparameters** (such as regularization, stability weight, integer proximity strength, etc.). For example, you can configure parameters to encourage "step" parameters to converge to integer values.
  * **Output Results:** The node provides both **raw floating-point numbers** and **rounded integer values**, allowing you to apply them directly based on your actual needs.
  * **Applicable Scenarios:** When the basic version cannot meet your precision requirements, or when you need more stringent control over the characteristics of the solution (e.g., forced integer values, smoother convergence), this advanced version should be prioritized.