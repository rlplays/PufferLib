#include "grid_interact.h"

#define Env GridInteractEnv
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_rewards = unpack(kwargs, "num_rewards");
    env->fov = unpack(kwargs, "fov");
    env->cell_types = unpack(kwargs, "cell_types");
    env->cell_size = unpack(kwargs, "cell_size");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
