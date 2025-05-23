# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
