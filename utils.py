# utils.py


def get_bicgDict():
    bicg_mat = {
    "f_name" : [],
    "bicg_iter" : [],

    "restart" : [],
    "no_restart" : [],
    "no_restart_1f" : [],
    "no_restart_2f" : [],

    "p_forward_max"  : [],
    "p_forward_min"  : [],
    "p_forward_mean" : [],
    "p_backward_max"  : [],
    "p_backward_min"  : [],
    "p_backward_mean" : [],

    "r_forward_max"  : [],
    "r_forward_min"  : [],
    "r_forward_mean" : [],
    "r_backward_max"  : [],
    "r_backward_min"  : [],
    "r_backward_mean" : [],

    "r_old_forward_max"  : [],
    "r_old_forward_min"  : [],
    "r_old_forward_mean" : [],
    "r_old_backward_max"  : [],
    "r_old_backward_min"  : [],
    "r_old_backward_mean" : [],

    "s_forward_max"  : [],
    "s_forward_min"  : [],
    "s_forward_mean" : [],
    "s_backward_max"  : [],
    "s_backward_min"  : [],
    "s_backward_mean" : [],

    "sigma_forward_max"  : [],
    "sigma_forward_min"  : [],
    "sigma_forward_mean" : [],
    "sigma_backward_max"  : [],
    "sigma_backward_min"  : [],
    "sigma_backward_mean" : [],

    "v_forward_max"  : [],
    "v_forward_min"  : [],
    "v_forward_mean" : [],
    "v_backward_max"  : [],
    "v_backward_min"  : [],
    "v_backward_mean" : [],

    "r_0_forward_max"  : [],
    "r_0_forward_min"  : [],
    "r_0_forward_mean" : [],
    "r_0_backward_max"  : [],
    "r_0_backward_min"  : [],
    "r_0_backward_mean" : [],

    "alpha_forward_max"  : [],
    "alpha_forward_min"  : [],
    "alpha_forward_mean" : [],
    "alpha_backward_max"  : [],
    "alpha_backward_min"  : [],
    "alpha_backward_mean" : [],

    "t_forward_max"  : [],
    "t_forward_min"  : [],
    "t_forward_mean" : [],
    "t_backward_max"  : [],
    "t_backward_min"  : [],
    "t_backward_mean" : [],

    "beta_forward_max"  : [],
    "beta_forward_min"  : [],
    "beta_forward_mean" : [],
    "beta_backward_max"  : [],
    "beta_backward_min"  : [],
    "beta_backward_mean" : [],

    "omega_forward_max"  : [],
    "omega_forward_min"  : [],
    "omega_forward_mean" : [],
    "omega_backward_max"  : [],
    "omega_backward_min"  : [],
    "omega_backward_mean" : [],

    "grad_norm" : []
    }

    return bicg_mat

def get_dfStencil():

    df_stencils = {
        "iter" : [],
        "f_name" : [],

        "d1_forward_max"  : [],
        "d1_forward_min"  : [],
        "d1_forward_mean" : [],
        "d1_backward_max"  : [],
        "d1_backward_min"  : [],
        "d1_backward_mean" : [],

        "d2_forward_max"  : [],
        "d2_forward_min"  : [],
        "d2_forward_mean" : [],
        "d2_backward_max"  : [],
        "d2_backward_min"  : [],
        "d2_backward_mean" : [],

        "boo_forward_max"  : [],
        "boo_forward_min"  : [],
        "boo_forward_mean" : [],
        "boo_backward_max"  : [],
        "boo_backward_min"  : [],
        "boo_backward_mean" : [],

        "bmo_forward_max"  : [],
        "bmo_forward_min"  : [],
        "bmo_forward_mean" : [],
        "bmo_backward_max"  : [],
        "bmo_backward_min"  : [],
        "bmo_backward_mean" : [],

        "bop_forward_max"  : [],
        "bop_forward_min"  : [],
        "bop_forward_mean" : [],
        "bop_backward_max"  : [],
        "bop_backward_min"  : [],
        "bop_backward_mean" : [],

        "bpo_forward_max"  : [],
        "bpo_forward_min"  : [],
        "bpo_forward_mean" : [],
        "bpo_backward_max"  : [],
        "bpo_backward_min"  : [],
        "bpo_backward_mean" : [],

        "bom_forward_max"  : [],
        "bom_forward_min"  : [],
        "bom_forward_mean" : [],
        "bom_backward_max"  : [],
        "bom_backward_min"  : [],
        "bom_backward_mean" : [],

        "grad_norm" : []
        }


    return df_stencils

