
# to install - run this command
# chmod 777 default_script.sh && ./default_script.sh

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	apt="sudo apt-get -y"
	sudo apt-get update
	sudo apt-get -y install zsh
	sudo apt-get -y install build-essential
	sudo apt-get -y install clang-tools-9
elif [[ "$OSTYPE" == "darwin"* ]]; then
	apt="brew"

	# install homebrew
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

	# install packages
	brew install iterm2
fi

$apt install vim
$apt install git
$apt install wget
$apt install make
$apt install gpg
$apt install python3
$apt install gdb
git clone https://github.com/longld/peda.git ~/peda
echo "source ~/peda/peda.py" >> ~/.gdbinit

sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.zsh-syntax-highlighting
wget https://raw.githubusercontent.com/tlucanti/little-projects/master/Config/.vimrc -O ~/.vimrc
wget https://raw.githubusercontent.com/tlucanti/little-projects/master/Config/.zshrc -O ~/.zshrc


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sudo apt-get install python-pip
	sudo apt-get install python3-pip
	sudo apt-get update
	sudo apt-get dist-upgrade
fi

ssh-keygen
gpg --full-generate-key
