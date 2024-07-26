import warnings

import awkward
from dask_awkward import dask_property

from coffea.nanoevents.methods import base, candidate, vector

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)

class _NanoAODEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return f"<event {getattr(self,'run','??')}:\
                {getattr(self,'luminosityBlock','??')}:\
                {getattr(self,'event','??')}>"


behavior["NanoEvents"] = _NanoAODEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    # behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class PtEtaPhiMCollection(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """Generic collection that has Lorentz vector properties"""

    pass


@awkward.mixin_class(behavior)
class DisMuon(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """NanoAOD muon object"""
    @dask_property
    def matched_gen(self):
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    @matched_gen.dask
    def matched_gen(self, dask_array):
        return dask_array._events().GenPart._apply_global_index(dask_array.genPartIdxG)
_set_repr_name("DisMuon")
