from PySide6.QtWidgets import QDoubleSpinBox

class DoubleSpinBox(QDoubleSpinBox):
  def __init__(self, value=0, min_val=0, max_val=1, decimals=3, single_step=0.001, *args, **kwargs):
    super().__init__(*args, **kwargs)
    value = max(value, min_val)
    value = min(value, max_val)

    self.setRange(min_val, max_val)
    self.setValue(value)
    self.setDecimals(decimals)
    self.setSingleStep(single_step)
    self.setButtonSymbols(self.buttonSymbols().NoButtons)
