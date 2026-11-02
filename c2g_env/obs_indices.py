"""
c2g_env/obs_indices.py  —  Centralised Observation Index Maps
==============================================================
Single source of truth for observation-vector indices in both
``C2GFastEnv`` (18-D) and ``C2GMacroEnv`` (19-D).

Import the namespace you need::

    from c2g_env.obs_indices import Fast, Macro

Then use e.g. ``obs[Fast.TEMP_A]`` or ``obs[Macro.LMP]``.
"""


class Fast:
    """Observation indices for C2GFastEnv (18-D)."""

    TEMP_A       = 0   # T_A / T_safe
    TEMP_B       = 1   # T_B / T_safe
    SOC          = 2   # BESS state-of-charge fraction
    P_BASE       = 3   # p_base_kw / FACILITY_CAP_KW
    P_FLEX       = 4   # p_flex_nom_kw / FACILITY_CAP_KW
    P_FAC        = 5   # p_facility_mw * 1000 / FACILITY_CAP_KW
    REGD         = 6   # clip(regd_signal, -1, 1)
    LMP          = 7   # lmp_usd_mwh / 200
    GRID_LOAD    = 8   # load_norm
    IS_SPIKE     = 9   # is_spike_active
    PREV_THR     = 10  # previous throttle
    PREV_PMP     = 11  # previous pump speed
    PUE          = 12  # pue_dynamic / 2.5
    T_AMB        = 13  # weather.temp_norm(tick)
    FREQ_DEV     = 14  # clip((f_grid - f_nom) / 0.5, -1, 1)
    VPCC         = 15  # clip(v_pcc_pu, 0, 1.1)
    BACKLOG      = 16  # backlog_kw / p_flex_max_kw
    COMMITTED    = 17  # committed_mw / committed_mw_max

    DIM = 18


class Macro:
    """Observation indices for C2GMacroEnv (19-D)."""

    TEMP_A       = 0   # temp_A_mean (normalised)
    TEMP_B       = 1   # temp_B_mean (normalised)
    SOC          = 2   # bess_soc_end
    P_BASE       = 3   # p_base_mean
    P_FAC        = 4   # p_facility_mean
    REGD         = 5   # |regd| mean
    LMP          = 6   # lmp_mean
    GRID_LOAD    = 7   # grid_load_mean
    TRACKING_ERR = 8   # tracking_err_mean
    IS_SPIKE     = 9   # is_spike_any
    HEADROOM_A   = 10  # thermal headroom zone A
    HEADROOM_B   = 11  # thermal headroom zone B
    BID_MW_PREV  = 12  # previous bid MW (normalised)
    BESS_PREV    = 13  # previous BESS / bid-price action
    FREQ_DEV     = 14  # freq_dev_mean
    VPCC         = 15  # v_pcc_mean
    BACKLOG      = 16  # backlog_norm_mean
    RMCP         = 17  # rmcp_norm
    REG_NEED     = 18  # reg_need_norm

    DIM = 19
