import psutil
import os
import socket
import pickle
import time
import awkward as ak
import numpy as np
import torch


class SimpleWorkLog:
    def __init__(self, label=""):
        """Create a worklog which stores basic info, most importantly a starting and ending timestamp and a label, as well as 
        info identifying the worker it was produced on."""
        self.label = label
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.start_time = None
        self.end_time = None
        self.interval = None
        self.model = None
        self.version = None
        self.batchsize = []
        self.bytesize = []
        
    def start(self, model_name=None, model_version=None):
        # Store a timestamp, as well as start a high-precision counter for the interval.
        self.start_time = time.gmtime()
        self.interval = time.perf_counter()
        # If an inference request, also store the model name and version
        self.model=model_name
        self.model_version=model_version
        
    def log_inference(self, batchsize=None, bytesize=None):
        # If we're doing inference, store the batchsize and bytesize for stat analysis later (multiple batches per WorkLog are expected)
        self.batchsize.append(batchsize)
        self.bytesize.append(bytesize)
        
    def end(self):
        # Finish the time-interval, and add another timestamp for redundancy
        self.interval = time.perf_counter() - self.interval
        self.end_time = time.gmtime()
        
def generate_pseudodata_from_seed(seed, chunksize=20000):
    """
    Parameters:
    seed (int): A seed for pseudo-random number generator
    chunksize (int): [Optional] Number of pseudo-events to generate, for each of which 1-10 pseudo 'jets'
    and associated data will be generated

    Returns:
    record_array (awkward.Array): An awkward record array with 3 fields (points, features, mask)
    
    Create data in a format analogous to High Energy Physics data as processed with the scikit-hep ecosystem.
    This is a stand-in for loading data stored in .root files via coffea
    """
    grdn = np.random.default_rng(seed)
    njets = ak.Array(grdn.integers(1, 10, size=chunksize))
    total_jets = np.sum(njets)    
    #points = ak.unflatten(grdn.random((total_jets, 2, 100), dtype=np.float32), njets)
    #features = ak.unflatten(grdn.random((total_jets, 5, 100), dtype=np.float32), njets)
    #mask = ak.unflatten(grdn.random((total_jets, 1, 100), dtype=np.float32), njets)
    return ak.unflatten(ak.Array({
        "points": grdn.random((total_jets, 2, 100), dtype=np.float32),
        "features": grdn.random((total_jets, 5, 100), dtype=np.float32),
        "mask": grdn.random((total_jets, 1, 100), dtype=np.float32)
    }),njets)

def run_inference_pnmodel(record_array, model, batchsize=1024, triton=False, worklog=SimpleWorkLog):
    """
    Parameters:
    record_array (awkward.Array): An awkward record array with 3 fields (points, features, mask)
    batchsize (int): Desired batchsize
    triton (bool): [Default: False] return data as a dict for triton inference, else a list for local inference
    
    Returns:
    inference_outputs (awkward.Array): An awkward array of inference results
    worklogs (list): List of worklogs tracking stats about the inference
    errors (None | list): List of triton inference errors, if any
    
    Restructure data from scikit-hep format to one suitable for inference (batched) with triton or local ParticleNet,
    and perform the inference
    """
    counts, flattened_record = ak.num(record_array), ak.flatten(record_array)
    total_records = len(flattened_record)
    outputs = []
    worklogs = []
    errors = []
    for start in range(0, total_records, batchsize):
        stop = min(total_records, start + batchsize)
        #try:
        #    dask.distributed.get_worker().log_event("run_inference_pnmodel", {
        #        "triton": triton,
        #        "batchsize": batchsize,
        #        "len": total_records,
        #        "progress": start/total_records
        #    })
        #except:
        #    pass
        nbytes = 0
        if triton:
            # Restructure
            worklogs.append(SimpleWorkLog("RestructureInputs"))
            worklogs[-1].start()
            X = dict()
            X["points__0"] = ak.to_numpy(flattened_record["points", start:stop]);
            X["features__1"] = ak.to_numpy(flattened_record["features", start:stop])
            X["mask__2"] = ak.to_numpy(flattened_record["mask", start:stop])
            # Calculate the bytes of the numpy arrays sent for the purpose of analyzing data sent
            for val in X.values():
                nbytes += val.nbytes
            worklogs[-1].end()
            
            worklogs.append(SimpleWorkLog("Inference"))
            worklogs[-1].start(model._model, model._version)
            worklogs[-1].log_inference(stop-start, nbytes)
            temp = model(X)
            if isinstance(temp, np.ndarray):
                outputs.append(temp)
            else:
                #Could make the output array Option type and fill with Nones instead of -1
                outputs.append(-1*np.ones_like(X["points__0"], dtype=np.float32))
                errors.append((start, stop, temp))
            worklogs[-1].end()
        else:
            # Restructure
            worklogs.append(SimpleWorkLog("RestructureInputs"))
            worklogs[-1].start()
            X = []
            for field in ["points", "features", "mask"]:
                X.append(torch.from_numpy(ak.to_numpy(flattened_record[field, start:stop])))        
            for val in X:
                nbytes += val.nelement() * val.element_size()
            worklogs[-1].end()
            
            #Run Inference
            with torch.no_grad():
                worklogs.append(SimpleWorkLog("Inference"));
                worklogs[-1].start("pn_demo", "1") #FIXME Add model/version dynamically
                worklogs[-1].log_inference(stop-start, nbytes)
                try:
                    temp = model(*X).detach().numpy()
                    outputs.append(temp)
                except Exception as inst:
                    outputs.append(-1*np.ones_like(X[0], dtype=np.float32))
                    errors.append((start, stop, inst))
                worklogs[-1].end()
    output = np.concatenate(outputs, axis=0)
    flattened_record = ak.with_field(flattened_record, output, "output")
    return ak.unflatten(flattened_record, counts), worklogs, errors
    
def get_triton_client(model_and_version="pn_demo/1", server="triton+grpc://triton.apps.okddev.fnal.gov:443/"):
    """
    Parameters:
    model_and_version (str): Name and version of the desired model on the triton server (e.g. 'pn_demo/1')
    server (str): protocol and address of server (e.g. 'triton+grpc://triton.apps.okddev.fnal.gov:443/')
    
    Returns:
    triton_model_client (triton client): A client capable of running inference on 
    
    Sets up the triton client, ready to perform inference requests
    """
    from utils.tritonutils import wrapped_triton

    # create instance of triton model
    triton_model = wrapped_triton(server + model_and_version)
    return triton_model

def create_local_pnmodel(dev="cpu"):
    from models.ParticleNet import ParticleNetTagger
    import torch

    # load in local model
    local_model = ParticleNetTagger(5, 2,
                            [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                            [(256, 0.1)],
                            use_fusion=True,
                            use_fts_bn=False,
                            use_counts=True,
                            for_inference=False)

    LOCAL_PATH = "/srv/models/pn_demo.pt"
    local_model.load_state_dict(torch.load(LOCAL_PATH, map_location=torch.device(dev)))
    local_model.eval()
    return local_model

def process_function(seed, chunksize=1000, batchsize=1024, triton=False, worklog=SimpleWorkLog, triton_server=None):
    """
    seed (int): A seed for pseudo-random data generation
    chunksize (int): A number of pseudo-'events' to generate, for each of which a random number of sub-objects are created
    batchsize (int): Number of inference requests in a batch. When smaller than the sum of objects across all events (~5.5*chunksize), is done iteratively
    triton (bool): Flat for whether to perform local inferene or Triton inference
    worklog (function): A worklog function with a 'start(model_name=None, model_version=None)', 'end()', and 'log_inference(batchsize, numbytes)' functions
    triton_server (http address): Address of the triton_server to pass to get_triton_client [Default: triton+grpc://triton.apps.okddev.fnal.gov:443/]
    """
    #import importlib
    #from utils.mlbench import SimpleWorkLog as worklog
    worklogs = []
    
    # Generate the Pseudodata
    worklogs.append(worklog(label="GeneratePseudodata"))
    worklogs[-1].start()
    inputs = generate_pseudodata_from_seed(seed, chunksize)
    worklogs[-1].end()
    
    # Create Model and run restructuring and inference
    local_model = None
    triton_model = None
    errors = None
    if triton:
        triton_model = get_triton_client() if triton_server is None else get_triton_client(server=triton_server)
        with_outputs, inf_worklogs, errors = run_inference_pnmodel(
            inputs, 
            triton_model, 
            batchsize=batchsize, 
            triton=True, 
            worklog=SimpleWorkLog
        )
    else:
        local_model = create_local_pnmodel()
        with_outputs, inf_worklogs, errors = run_inference_pnmodel(
            inputs, 
            local_model, 
            batchsize=batchsize, 
            triton=False, 
            worklog=SimpleWorkLog
        )
    # Add with the worklogs for return to the original working client
    worklogs += inf_worklogs
    
    return {"worklogs": worklogs, 
            "output_desc": {"type": with_outputs.type, 
                            "fields": with_outputs.fields,
                            "layout": str(with_outputs.layout),
                            "disc_mean": np.mean(with_outputs.output),
                           },
            "triton_errors": errors,
           }
    
#Additional models to test
# https://github.com/suyong-choi/ABCDnn
# Particle Transformer
# Moderately complex BDT