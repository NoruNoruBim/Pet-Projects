class Logger(object):
    def __init__(self, filename='log.txt'):
        self.filename = filename


    def write_to_log(self, info):
        with open('log_folder\\' + self.filename, 'a') as f:
            f.write("\n---\n" + info)        


    def shrink_log(self, num=80000):
        self.write_to_log("shrink_log")
        with open('log_folder\\' + self.filename, 'r') as f:
            text = f.read().split('\n')
        self.write_to_log(str(len(text)))
        if num < len(text):
            with open('log_folder\\' + self.filename, 'w') as f:
                for line in text[-num:]:
                    f.write(line + '\n')
        self.write_to_log(str(min(len(text), num)))
