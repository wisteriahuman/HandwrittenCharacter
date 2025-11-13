import torch
from torchvista import trace_model
from models.simple_cnn import Simple_CNN
import os
import glob
import atexit
import signal
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

generated_files = []

def cleanup():
    for file in generated_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def signal_handler(sig, frame):
    print("\n\nサーバーを停止中...")
    cleanup()
    sys.exit(0)

def create_clickable_link(url):
    return f"\033]8;;{url}\033\\{url}\033]8;;\033\\"

def main():
    model = Simple_CNN()
    model.eval()
    sample_input = torch.rand(1, 1, 28, 28)
    
    before = set(glob.glob("torchvista_graph_*.html"))
    trace_model(model, sample_input, export_format="html")
    after = set(glob.glob("torchvista_graph_*.html"))
    new_files = after - before
    generated_files.extend(new_files)
    
    port = 8000
    
    if new_files:
        html_file = list(new_files)[0]
        url = f"http://localhost:{port}/{html_file}"
        
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        print("\n" + "=" * 60)
        print(f"{GREEN}Model Visualization Server{RESET}")
        print()
        print(f"  {BOLD}➜{RESET}  Local:   {CYAN}{create_clickable_link(url)}{RESET}")
        print()
        print(f"  Press {BOLD}Ctrl+C{RESET} to stop")
        print("=" * 60 + "\n")
    else:
        return
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    
    server = HTTPServer(('', port), SimpleHTTPRequestHandler)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()

if __name__ == "__main__":
    main()
