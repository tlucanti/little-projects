filetype plugin indent on
" Включает определение типа файла, загрузку
" соответствующих ему плагинов и файлов отступов

set encoding=utf-8
" Ставит кодировку UTF-8

set nocompatible
" Отключает обратную совместимость с Vi

syntax enable
" Включает подсветку синтаксиса

" set expandtab
" Замена табов на пробелы

set smarttab
" авто выставление табов в начале строки

set tabstop=4
" размер таба при печати

set softtabstop=4
" размер таба при удалении

set shiftwidth=4
" ширина табов в пробелах

set number
" абсолютная нумерация строк

"set relativenumber
" относительная нумерация строк

" set foldcolumn=2
" дополнительный отступ слева

set mouse=a
" включение полной поддержки мыши

set ignorecase
" игнорирование регистра при поиске

set smartcase

set hlsearch
" подсветка при поиске

set incsearch
" подсказка первого вхождения при поиске

set whichwrap+=h,l,<,>,[,]
" авто перенос курсора на следующую строку

set clipboard=unnamedplus
" копирование в системный буффер вместо внутреннего вима

map <C-c> y
" дополнительный шорткат на копирование на CTRL+C

set rtp+=$HOME/.local/lib/python2.7/site-packages/powerline/bindings/vim/

" Always show statusline
set laststatus=2

" Use 256 colours (Use this setting only if your terminal supports 256 colours)
" set t_Co=256

set colorcolumn=81

if empty(glob('~/.vim/autoload/plug.vim'))
  silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim 
  " Если vim-plug не стоит
  " Создать директорию
  " И скачать его оттуда
  " А после прогнать команду PlugInstall
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

call plug#begin('~/.vim/bundle')
" Начать искать плагины в этой директории
" Тут будут описаны наши плагины

    Plug 'ErichDonGubler/vim-sublime-monokai'
    Plug 'octol/vim-cpp-enhanced-highlight'
    Plug 'preservim/nerdtree'
    Plug 'frazrepo/vim-rainbow'
	" Plug 'Yggdroot/indentLine'
	" Plug 'lukas-reineke/virt-column.nvim'
call plug#end()
" Перестать это делать

" vim rainbow settings
au FileType c,cpp,py,sh,vimrc,zshrc call rainbow#load()
let g:rainbow_active = 1
let g:rainbow_ctermfgs = ['lightblue', 'red', 'lightgreen', 'yellow', 'magenta']

" IndentLinw settings
let g:indentLine_enabled = 1
let g:indentLine_char_list = ['│', '╎', '┆', '┊']
" let g:indentLine_color_term = 'lightgrey'

" colorscheme settings
colorscheme sublimemonokai
hi Normal guibg=NONE ctermbg=NONE
"disabling colorscheme bg color (to transparent)

