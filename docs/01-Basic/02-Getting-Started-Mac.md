<style>
.custom {
    background-color: #008d8d;
    color: white;
    padding: 0.25em 0.5em 0.25em 0.5em;
    white-space: pre-wrap;       /* css-3 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;
}

pre {
    background-color: #027c7c;
    padding-left: 0.5em;
}

</style>

# Getting Started on Mac

- Author: [JeongHo Shin](https://github.com/ThePurpleCollar)
- Peer Review: [JeongGi Park](https://www.linkedin.com/in/jeonggipark/), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/02-Getting-Started-Mac.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/02-Getting-Started-Mac.ipynb)
## Overview

This guide provides a comprehensive setup process tailored for developing with LangChain on a Mac. LangChain is a framework for building applications powered by large language models (LLMs), and this guide ensures your environment is fully optimized for seamless integration and development.

# Table of Contents

- [Overview](#overview)
- [Opening Terminal](#opening-terminal)
- [Installing Homebrew](#installing-homebrew)
- [Verifying Xcode Installation](#verifying-xcode-installation)
- [Downloading Practice Code](#downloading-practice-code)
- [Python and Environment Configuration](#python-and-environment-configuration)
- [Development Tools Setup](#development-tools-setup)

----

## Opening Terminal
- Open Spotlight Search by pressing **Command + Space** .

- Search for **terminal** and press Enter to open the Terminal.

## Installing Homebrew

### Running the Homebrew Installation Command
- Run the following command in the Terminal to install Homebrew:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

- Enter your account password when prompted.
![PasswordZ](./img/01-Follow-the-Installation-Video_Mac-01.png)
     
- Press ENTER to proceed with the installation.

### Configuring Homebrew Environment

- Run the following command to check your username:
   ```bash
   
   whoami
   ```
![Jupyter Extension](./img/01-Follow-the-Installation-Video_Mac-02.png)

- Check the installation path of Homebrew:
   ```bash

   which brew
   ```

- Verify the installation path of Homebrew:
   - **Case 1** : If the output is **/opt/homebrew/bin/brew** , use the following command to configure the environment:
      ```bash
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/<your-username>/.zprofile
      ```

   - **Case 2** : If the output is **/usr/local/bin/brew** , use the following command:
      ```bash
      echo 'eval "$(/usr/local/bin/brew shellenv)"' >> /Users/<your-username>/.zprofile
      ```

## Verifying Xcode Installation

To check if Xcode Command Line Tools are installed, run the following command in your terminal:

```bash
xcode-select --install
```


## Downloading Practice Code

[Reference] Practice code repository: [LangChain Practice Code](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)


### Verifying Git Installation

- Check if Git is installed by running the following command in your terminal:
   ```bash
   git --version
   ```

- If the command outputs the Git version, you already have Git installed, and no further action is required.

- If Git is not installed, you can install it using Homebrew:
   ```bash
   brew install git
   ```

- After installation, verify Git again:
   ```bash
   git --version
   ```



### Downloading Practice Code with Git
- Navigate to the **Documents** folder (or any other folder where you want to download the practice code). Use the following command:
   ```bash
   cd Documents
   ```
- If you want to use a different directory, replace Documents with your desired path.

- Use the **git** command to download the practice code from the repository. Run the following command in your terminal:
   ```bash
   git clone https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial.git
   ```
![](./img/01-Follow-the-Installation-Video_Mac-03.png)

- The repository will be cloned into a folder named LangChain-OpenTutorial within the selected directory.


## Installing Pyenv

#### Reference
For detailed documentation, refer to the [Pyenv GitHub Page](https://github.com/pyenv/pyenv?tab=readme-ov-file#understanding-python-version-selection).

---

#### Steps to Install Pyenv

1. Update Homebrew and install ```pyenv``` using the following commands:
   ```bash
   brew update
   brew install pyenv
   ```

2. Add the following lines to your ~/.zshrc file. Copy and paste the commands into your terminal:
   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   ```

3. If you encounter a permissions error, resolve it by running these commands:
   ```bash
   sudo chown $USER ~/.zshrc
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   ```

4. Restart the terminal shell to apply the changes:
   ```bash
   exec "$SHELL"
   ```


## Installing Python

- Use ```pyenv``` to install Python 3.11:
   ```bash
   pyenv install 3.11
   ```

- Set Python 3.11 as the global Python version:
    ```bash
    pyenv global 3.11
    ```

- Restart the shell to ensure the changes take effect:
    ```bash
    exec zsh
    ```

- Verify the installed Python version:
    ```bash
    python --version
    ```

- Ensure the output shows 3.11.

## Installing Poetry

#### Reference
For detailed documentation, refer to the [Poetry Official Documentation](https://python-poetry.org/docs/#installing-with-the-official-installer).

---

#### Steps to Install and Configure Poetry

- Install Poetry using ```pip3```:
   ```bash
   pip3 install poetry
   ```

- Set up a Python virtual environment using Poetry:
   ```bash
    poetry shell
    ```

- Update all Python dependencies in the project:
   ```bash
    poetry update
    ```




## Installing Visual Studio Code

- **Download Visual Studio Code**:
   - Visit the [Visual Studio Code Download Page](https://code.visualstudio.com/download).
   - Download the installer for your operating system.

- **Install Visual Studio Code**:
   - Follow the installation instructions for your system.
   - Dag the application to the **Applications** folder.

- **Install Extensions**:
   - Open Visual Studio Code.
   - Click on the **Extensions** icon on the left sidebar.
![Extensions Icon](./img/01-Follow-the-Installation-Video_Mac-04.png)

   - Search for **"python"** in the Extensions Marketplace and install it.
![Python Extension](./img/01-Follow-the-Installation-Video_Mac-05.png)

   - Search for **"jupyter"** in the Extensions Marketplace and install it.
![Jupyter Extension](./img/01-Follow-the-Installation-Video_Mac-06.png)

- **Restart Visual Studio Code**:
   - After installing the extensions, restart Visual Studio Code to apply the changes.

- **Select Python Environment**:
   - Click on **"Select Kernel"** in the top-right corner of Visual Studio Code.
   - Choose the Python virtual environment you set up earlier.

   - **Note**: If your environment does not appear in the list, restart Visual Studio Code.

---

Now, Visual Studio Code is fully set up and ready for development with Python and Jupyter support.



