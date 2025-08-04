/* Pure C demo file for grid_interact. Build it with:
 * bash scripts/build_ocean.sh grid_interact local (debug)
 * bash scripts/build_ocean.sh grid_interact fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#define ALLOW_LOGGING 1
#include "grid_interact.h"
#include "puffernet.h"

int main(int argc, char **argv) {
  GridInteractEnv env = {.width = 1200,
                         .height = 1200,
                         .cell_size = 100,
                         .fov = 6,
                         .num_rewards = 15,
                         .cell_types = NUM_CELL_TYPES, // W, #, R, G, EMPTY, P
                         .ego_centric_view = true};

  // Helps keep the number of observations constant regardless of the number of
  // agents/goals/rewards etc.
  int num_obs = get_num_obs(&env);

  int logit_sizes[1] = {5};

  bool use_trained_model = argc > 1 && strcmp(argv[1], "trained") == 0;

  // Weights are exported by running puffer export
  Weights *weights = NULL;
  LinearLSTM *net = NULL;
  if (use_trained_model) {
    weights = load_weights("resources/grid_interact/grid_interact_weights.bin",
                           299654);
    net = make_linearlstm(weights, 1, num_obs, logit_sizes, 1);
  }

  allocate(&env, use_trained_model);

  // Always call reset and render first
  c_reset(&env);
  c_render(&env);

  int frame_index = 0;
  while (!WindowShouldClose()) {
    if (use_trained_model) {
      // Only run the model at a lower fps to give the user a chance to react.
      if (frame_index % 4 == 0) {
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
  }
  free_allocated(&env);
}
