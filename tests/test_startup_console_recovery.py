from src import run_bot


def test_startup_recovery_summary_is_compact():
    summary = run_bot.format_startup_recovery_summary(
        [
            {
                'order_id': 1,
                'status': 'not_found_on_venue',
                'result': {
                    'order': {'id': 1, 'status': 'not_found_on_venue'},
                    'response': {'raw': {'result': None, 'nested': {'too': 'noisy'}}},
                },
            },
            {'order_id': 2, 'status': 'open', 'result': {'response': {'raw': {'huge': 'blob'}}}},
            {'order_id': 3, 'status': 'canceled'},
            {'order_id': 4, 'status': 'unknown'},
        ]
    )

    assert summary == 'Startup order recovery | total=4 recovered=1 canceled=1 not_found=1 still_unknown=1'
    assert 'nested' not in summary
    assert 'huge' not in summary
