## Overview

A agent/bot to monitor the crypto market, send alerts when a token is oversold (oportunity) and open/close orders in a perpetuals exchange via commands on telegram bot

Project in development phase...

## How to Run Locally

1.  **Create and activate a virtual environment:**
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```sh
    python src/main.py
    ```

## How to Run with Docker

1.  **Create the Environment File:**
    Before running with Docker, you need to create a `.env` file in the root of the project. This file will hold your secret keys.

    Create a file named `.env` and add the following content, replacing the placeholder values with your actual credentials:
    ```
    COINGECKO_API_KEY_ACCOUNT=YOUR_COINGECKO_API_KEY
    TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
    ```

2.  **Build the Docker image:**
    ```sh
    docker build -t trading-bot .
    ```

3.  **Run the container in the foreground:**
    This command runs the container, securely injects your secrets from the `.env` file, and streams logs to your terminal. Press `Ctrl+C` to stop. The `--rm` flag automatically removes the container when it exits.
    ```sh
    docker run --rm --env-file .env trading-bot
    ```

4.  **Run the container in the background (detached mode):**
    To run the application continuously on a server, you can start it in detached mode.
    ```sh
    docker run -d --name trading-bot-instance --restart always --env-file .env trading-bot
    ```

5.  **View logs for the background container:**
    ```sh
    docker logs -f trading-bot-instance
    ```

6.  **Stop the background container:**
    ```sh
    docker stop trading-bot-instance
    ```
