# Telegram Bot BasilAI

## Install

- prod environment

    ```bash
    make install-prod
    ```

- dev environment

    ```bash
    make install-dev
    ```

## Generate dataset

```bash
make dataset RAW_DATA=<path to telegram dialogs dump json> TARGET=<user_id> DATASET=<path to csv file to save generated dataset>
```

## Train model

```bash
make train TRAIN_CONFIG=<path to config file>
```

## Run telegram bot locally

```bash
make bot TOKEN=<telegram bot auth token>
```

## Heroku hosting

### Getting started with heroku

- authenticate on heroku

  ```bash
  heroku login
  ```

- add git `heroku` remote

  ```bash
  heroku git:remote -a basilai-bot
  ```
  
### Pushing new version to heroku

- push local `master` branch to heroku `main` branch

  ```bash
  git push heroku master:main
  ```
  
  > _you can push any local branch - just change `master` to branch name_

- heroku restarts application with new version automatically
  
### Scaling heroku application

- check heroku running apps

  ```bash
  heroku ps
  ```

- scale heroku apps

  ```bash
  heroku ps:scale worker=1
  ```

- stop heroku apps

  ```bash
  heroku ps:stop worker.1
  ```

### Get logs

```bash
heroku logs -t
```