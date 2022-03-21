

class WhiteString():
    def __init__(self):
        self.string_position_list = []
        self.string_width_list = []

    def set(self, position_value, width_value ):
        self.string_position_list.append(position_value)
        self.string_width_list.append(width_value)
    def get(self, index):
        return self.string_position_list[index], self.string_width_list[index]