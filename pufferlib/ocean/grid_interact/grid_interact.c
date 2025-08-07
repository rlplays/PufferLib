/* Pure C demo file for grid_interact. Build it with:
 * bash scripts/build_ocean.sh grid_interact local (debug)
 * bash scripts/build_ocean.sh grid_interact fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#define ALLOW_LOGGING 1
#include "grid_interact.h"
#include "puffernet.h"
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    Config* config = load_config("resources/grid_interact/grid_interact_config.ini");
    assert(config != NULL);
    // Match the env used during training.
    GridInteractEnv env = {.width = config_getint(config, "env.width", 1600), 
                          .height = config_getint(config, "env.width", 1600), 
                          .cell_size = config_getint(config, "env.cell_size", 100), 
                          .fov = config_getint(config, "env.fov", 9), 
                          .num_rewards = config_getint(config, "env.num_rewards", 10), 
                           .cell_types = NUM_CELL_TYPES, // W, #, R, G, P (Exclude EMPTY)
                           .set_max_moves = 0};

    // Helps keep the number of observations constant regardless of the number of
    // agents/goals/rewards etc.
    int num_obs = get_num_obs(&env);

    int logit_sizes[1] = {5};

    bool use_trained_model = argc > 1 && strcmp(argv[1], "trained") == 0;

    // Weights are exported by running puffer export
    Weights *weights = NULL;
    LinearLSTM *net = NULL;
    if (use_trained_model) {
        weights = load_weights_from_config(config);
        net = make_linearlstm(weights, 1, num_obs, logit_sizes, 1);
    }

    allocate(&env, use_trained_model);

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    int frame_index = 0;
    if (strcmp(argv[1], "perf") == 0) {
      clock_t start = clock();
      double num_steps = 1000*1000;
      int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
      for (int i = 0; i < (int)num_steps; i++) {
        c_step(&env);
      }
      clock_t end = clock();
      double cpu_time_used_ms = ((double) (end - start) * 1000.0) / CLOCKS_PER_SEC;
      TLOG(LOG_INFO, "CPU time (%.0f steps): %0.9f ms per core\n", num_steps, cpu_time_used_ms);
      TLOG(LOG_INFO, "CPU time (%.0f steps): %0.9f ms\n", num_steps*1000.0, cpu_time_used_ms*1000.0);
      TLOG(LOG_INFO, "CPU time (1 step): %.9f ms\n", cpu_time_used_ms/num_steps);
      TLOG(LOG_INFO, "CPU time multi-threaded %d cores  (%.0f steps): %0.9f ms\n", num_cores, num_steps, (cpu_time_used_ms)/(double)num_cores);
      // AMD Ryzen TR 3970x (32c/64t) 1million steps: 393ms (unoptimized); 13ms (opt)
      // Theoretically, we should be able to run 100 million steps in 13 seconds on a 32 core CPU.
      // I wonder what the Python overhead is as we are running each step in a single thread, and transferring data
      // between C and Python.
    }
    while (!WindowShouldClose()) {
        if (use_trained_model) {
            // Only run the model at a lower fps to give the user a chance to react.
            if (frame_index % 10 == 0) {
                forward_linearlstm(net, env.observations, env.actions);
            }
        }
        c_step(&env);
        c_render(&env);
        ++frame_index;
    }

    // Try to clean up after yourself
    if (use_trained_model) {
        free_linearlstm(net);
        free(weights);
        free_config(config);
    }
    free_allocated(&env);
}
