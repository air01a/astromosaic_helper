import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

stop = False

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, callback, validator):
        self.validator=validator
        self.callback = callback

    def on_created(self, event):
        if event.is_directory:
            # Ignorer les événements liés aux nouveaux répertoires
            return

        # Le nouvel événement de fichier a été cr   éé
        if self.validator(event.src_path):
            print(f"Nouveau fichier créé: {event.src_path}")
            self.callback(event.src_path)

def stop_watch_directory():
    global stop
    stop = True

def watch_directory(path, callback, extensions):
    global stop
    event_handler = NewFileHandler(callback,extensions)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    return observer


if __name__ == "__main__":
    # Exemple d'utilisation
    directory_to_watch = "test"
    watch_directory(directory_to_watch)