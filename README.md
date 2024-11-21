# Serefin-Xloop
Code repository for the Serefin Project.

### Prerequisites
- Python 3.8 or higher
- Make (optional)

### Instructions

1. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Or using make:
   make install
   ```

3. **Run Script**
   ```bash
   python src/app.py
   # Or using make:
   make run
   ```

4. **Development Tools** (optional)
   Install development dependencies:
   ```bash
   pip install black flake8 pylint isort
   ```

   Format code:
   ```bash
   make format
   ```

   Run linting:
   ```bash
   make lint
   ```

5. **Clean Project**
   ```bash
   make clean
   ```
