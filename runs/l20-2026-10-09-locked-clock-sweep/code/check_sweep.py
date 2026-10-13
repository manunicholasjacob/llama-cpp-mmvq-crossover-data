#!/usr/bin/env python3
"""Offline contract checks. Synthetic rows are not GPU evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import sweep


def primary_rows(clocks, factor):
    rows = []
    for item in sweep.schedule(clocks):
        for n in item['ns']:
            gain = factor(item['clock'], item['quant']) if item['world'] == 'cutoff7' and n == 8 else 1
            rows.append({**item, 'n': n, 'avg_ts': 100 * gain})
    return rows


class Contracts(unittest.TestCase):
    def test_protocol_matches_runner(self):
        protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())
        self.assertEqual(tuple(protocol['grid_mhz']), sweep.GRID)
        self.assertEqual(tuple(protocol['high_candidates_mhz']), sweep.HIGH_CANDIDATES)
        self.assertEqual(protocol['fixed_commit'], sweep.FIXED)

    def test_schedule_rotates_and_keeps_n8_separate(self):
        clocks = [1200, 1500, 1800, 2100, 2400]
        rows = sweep.schedule(clocks)
        primary = [x for x in rows if x['phase'] == 'primary']
        controls = [x for x in rows if x['phase'] == 'route_control']
        self.assertEqual(len(primary), len(clocks) * 6 * 2 * 3)
        self.assertEqual(len(controls), 2 * 2 * 3)
        self.assertTrue(all(x['ns'] == [8] for x in primary))
        self.assertTrue(all(x['ns'] == [1, 16] for x in controls))
        self.assertEqual({x['clock'] for x in controls}, {1200, 2400})
        self.assertTrue(all(x['round'] == 0 for x in controls))
        for clock, quant, world in ((1200, 'Q4_0', 'selected'), (2400, 'Q8_0', 'cutoff7')):
            positions = [x['position'] for x in primary if (x['clock'], x['quant'], x['world']) == (clock, quant, world)]
            self.assertEqual(sorted(positions), [0, 0, 1, 1, 2, 2])

    def test_material_rule_requires_three_percent_and_six_rounds(self):
        clocks = [1200, 1500, 2100]

        def factor(clock, quant):
            if clock == 1200:
                return 1.12
            if clock == 1500 and quant == 'Q4_0':
                return 1.05
            if clock == 1500:
                return 1.02
            return 0.85

        cells = {(x['clock'], x['quant']): x for x in sweep.reduce_primary(primary_rows(clocks, factor), clocks)}
        self.assertTrue(cells[1200, 'Q4_0']['material_win'])
        self.assertTrue(cells[1500, 'Q4_0']['material_win'])
        self.assertFalse(cells[1500, 'Q8_0']['material_win'])
        self.assertTrue(cells[2100, 'Q4_0']['material_loss'])
        reports = {quant: sweep.crossover([x for x in cells.values() if x['quant'] == quant]) for quant in ('Q4_0', 'Q8_0')}
        self.assertEqual(reports['Q4_0']['status'], 'CROSSED')
        self.assertEqual((reports['Q4_0']['last_win_mhz'], reports['Q4_0']['first_nonwin_mhz']), (1500, 2100))
        self.assertEqual((reports['Q8_0']['last_win_mhz'], reports['Q8_0']['first_nonwin_mhz']), (1200, 1500))
        self.assertEqual(sweep.boundary_clocks(reports, clocks), [1200, 1500, 2100])

    def test_split_rounds_and_bad_null_are_not_wins(self):
        clocks = [1200, 1500]
        rows = primary_rows(clocks, lambda clock, quant: 1.20)
        for row in rows:
            if row['world'] == 'cutoff7' and row['n'] == 8 and row['round'] == 3:
                row['avg_ts'] = 90
        self.assertFalse(any(x['material_win'] for x in sweep.reduce_primary(rows, clocks)))
        noisy = primary_rows(clocks, lambda clock, quant: 1.20)
        for row in noisy:
            if row['world'] == 'null_rebuild' and row['n'] == 8:
                row['avg_ts'] = 110
        self.assertTrue(all(not x['control_pass'] and not x['material_win'] for x in sweep.reduce_primary(noisy, clocks)))

    def test_upward_reversal_is_not_one_cutoff(self):
        clocks = [1200, 1500, 2100]

        def factor(clock, quant):
            return 0.9 if clock == 1500 else 1.1

        cells = sweep.reduce_primary(primary_rows(clocks, factor), clocks)
        self.assertEqual(sweep.crossover([x for x in cells if x['quant'] == 'Q4_0'])['status'], 'NONMONOTONIC')

    def test_choose_clocks_keeps_highest_held_lock(self):
        rows = [{'mhz': mhz, 'accepted': mhz != 1800 and mhz != 2520} for mhz in (1200, 1500, 1800, 2100, 2520, 2460)]
        self.assertEqual(sweep.choose_clocks(rows), [1200, 1500, 2100, 2460])
        with self.assertRaises(RuntimeError):
            sweep.choose_clocks([{'mhz': 1200, 'accepted': False}])

    def test_pin_tolerance_is_fifteen_megahertz(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'clock.csv'
            path.write_text('t, 300, 9001, 10, 250, 40, 0, 0\n' + 't, 1215, 9001, 50, 250, 50, 90, 0\n' * 4)
            self.assertTrue(sweep.read_clocks(path, 1200)['held'])
            path.write_text('t, 1216, 9001, 50, 250, 50, 90, 0\n' * 4)
            self.assertFalse(sweep.read_clocks(path, 1200)['held'])
            path.write_text('t, 1200, 9001, 50, 250, 50, 90, 0\n')
            with self.assertRaises(RuntimeError):
                sweep.read_clocks(path, 1200)

    def test_analysis_accepts_a_clean_curve_and_stops_on_heat(self):
        clocks = [1200, 1500, 2100]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / 'raw').mkdir()
            (folder / 'probe.json').write_text(json.dumps({'accepted_mhz': clocks}))
            rows = primary_rows(clocks, lambda clock, quant: 1.10 if clock < 2100 else 0.85)
            for row in rows:
                if row['n'] == 8:
                    row['telemetry'] = f'raw/{row["clock"]}.csv'
            (folder / 'raw/bench.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            for mhz in clocks:
                (folder / f'raw/{mhz}.csv').write_text(f't, {mhz}, 9001, 80, 275, 50, 90, 0\n' * 4)
            self.assertEqual(sweep.analyze(folder), 'CURVE_READY_FOR_REVIEW')
            report = json.loads((folder / 'analysis.json').read_text())
            self.assertEqual(report['crossover']['Q4_0']['status'], 'CROSSED')
            self.assertEqual(report['crossover']['Q4_0']['first_nonwin_mhz'], 2100)
            (folder / 'raw/2100.csv').write_text('t, 2100, 9001, 80, 275, 90, 90, 0\n' * 4)
            self.assertEqual(sweep.analyze(folder), 'INCONCLUSIVE_PIN_OR_CONTROL')


if __name__ == '__main__':
    unittest.main()
