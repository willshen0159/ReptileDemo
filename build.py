def write_file(src, dst):
    while True:
        line = src.readline()
        if(not line):
            break
        dst.write(line)
    return

with open("./build/webworker.js", "w") as wfp:
    with open("./deps/jsnet.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./deps/model.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./src/webworker.js", "r") as rfp:
        write_file(rfp, wfp)
        
with open("./build/app.js", "w") as wfp:
    with open("./src/drawing.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./src/evaluator.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./src/predictions.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./src/ui.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./src/default.js", "r") as rfp:
        write_file(rfp, wfp)
    with open("./deps/jsnet.js", "r") as rfp:
        write_file(rfp, wfp)
    wfp.write('\nvar WORKER_DATA = "')
    with open("./build/webworker.js", "r") as rfp:
        while True:
            line = rfp.readline()
            if(not line):
                break
            line = line[:-1]
            line = line + "\\n"
            wfp.write(line)
    wfp.write('";')
