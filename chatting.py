import sys

from setup import Options


if __name__ == "__main__":
    opt = Options()
    params = [len(opt.devices), len(opt.models)]
    pos = 0
    if len(sys.argv) == 3:
        for arg in sys.argv[1:]:
            val = arg
            while True:
                try:
                    val = int(val)
                    if 0 <= val < params[pos]:
                        params[pos] = val
                        pos += 1
                        break
                    else: val = input("Enter valid value: ")
                except:
                    val = input("Enter valid value: ")


