"""Route forecast() through the ensemble when replicates are configured."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")

old = '        if self.spatial_model == "particles":\n            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)\n        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)'

new = ('        if self.ensemble_replicates and self.ensemble_replicates > 1:\n'
       '            return self._forecast_ensemble(agents, grid, walls, gates, exits, sim_time_sec)\n'
       '        if self.spatial_model == "particles":\n'
       '            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)\n'
       '        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)')

n = text.count(old)
print(f"anchor occurrences: {n}")
if n:
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("dispatcher inserted")
print(f"forecast_ensemble references: {text.count('_forecast_ensemble')}")
