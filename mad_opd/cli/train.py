"""MAD-OPD training entrypoint.

Applies the MAD-OPD runtime patches (``mad_opd._bootstrap``) and then delegates
to ``ms-swift``'s own ``swift.cli.rlhf.main()`` so the full ``--rlhf_type gkd``
pipeline is reused unchanged.

Invoke via ``torchrun``::

    torchrun --nproc_per_node=2 --master_port=29500 \\
        -m mad_opd.cli.train \\
        --rlhf_type gkd --gkd_algorithm mad_opd \\
        --model <student>                --teacher_model <t1> \\
        --teacher_model_2 <t2>           --dataset <jsonl> \\
        --use_vllm_teachers true \\
        --vllm_teacher_1_url http://127.0.0.1:8001/v1 \\
        --vllm_teacher_2_url http://127.0.0.1:8002/v1 \\
        --report_to tensorboard          --output_dir ./outputs/run

The training flags are otherwise identical to those of ``swift rlhf``.
"""
import mad_opd  # noqa: F401 — installs patches on import


def main() -> None:
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()


if __name__ == '__main__':
    main()
