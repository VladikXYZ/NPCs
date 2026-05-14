from setup import Options

if __name__ == "__main__":
    opt = Options()
    # opt.devices_info()
    opt.select_device(0)
    opt.select_model(0)
    opt.run_llm()