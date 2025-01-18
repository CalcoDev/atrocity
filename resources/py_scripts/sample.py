import time
import sys
import threading

def main():
    cnt = 0
    stop_flag = [False]

    # Function to listen for 'q' from stdin
    def listen_for_input():
        for line in sys.stdin:
            if line.strip().lower() == 'q':
                stop_flag[0] = True
                break

    # Start the input listener in a separate thread
    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    # Main loop
    while not stop_flag[0]:
        time.sleep(0.5)
        cnt += 1
        print("counter is", cnt, flush=True)

    print("Finished!", flush=True)

if __name__ == "__main__":
    main()
