import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            # Ignorer les événements liés aux nouveaux répertoires
            return

        # Le nouvel événement de fichier a été créé
        print(f"Nouveau fichier créé: {event.src_path}")

def watch_directory(path):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    # Exemple d'utilisation
    directory_to_watch = "test"
    watch_directory(directory_to_watch)