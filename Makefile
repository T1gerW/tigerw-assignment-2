# Define the Python environment and application settings
VENV = venv
FLASK_APP = app.py
PORT = 3000

# Define commands
# Create a virtual environment and install dependencies
install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

# Run the Flask application on the specified port
run:
	. $(VENV)/bin/activate && \
	export FLASK_APP=$(FLASK_APP) && \
	export FLASK_ENV=development && \
	export FLASK_DEBUG=1 && \
	flask run --host=0.0.0.0 --port=$(PORT)

# Clean up the virtual environment
clean:
	rm -rf $(VENV)

