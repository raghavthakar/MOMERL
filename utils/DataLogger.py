import json
import nsgaii

class DataLogger:
    def __init__(self, target_filename):
        self.target_filename = target_filename

        # clear out the file
        with open(self.target_filename, 'w') as f:
            pass
    
    def straight_write(self, field_key, value):
        '''
        Append a new field and its value to the data file.
        '''
        write_data = {field_key : value}
        with open(self.target_filename, 'a') as f:
            json.dump(write_data, f, indent=4)
            f.write('\n')