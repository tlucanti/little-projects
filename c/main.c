#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <malloc.h>
#include <string.h>

//#define and &&
//#define or ||

typedef struct	s_symbol
{
    u_int8_t		sym;
    u_int32_t		cnt;
    char            *code;
}				t_symbol;

typedef struct	s_tree
{
    struct		s_tree *left;
    struct		s_tree *right;
    u_int32_t	data;
    char		c;
}				t_tree;

t_tree **archivate(t_symbol **hash);
void     write_next_char(int fd, char *code);

int 	read_file(const char *fname, t_symbol **hash, u_int32_t *sym_cnt)
{
    int			fd;
    u_int8_t	c;

    fd = open(fname, O_RDONLY);
    if (fd == -1)
        return (1);
    while (read(fd, &c, 1))
    {
        (hash[c]->cnt)++;
        (*sym_cnt)++;
    }
    close(fd);
    return (0);
}

void		sort(t_symbol **hash)
{
    t_symbol	*ptr;

    for (u_int16_t i=0; i < 256; ++i)
        for (u_int16_t j=1; j < 256; ++j)
            if (hash[j]->cnt > hash[j - 1]->cnt)
            {
                ptr = hash[j];
                hash[j] = hash[j - 1];
                hash[j - 1] = ptr;
            }
}

char    *append(char *str, char c) {
    size_t str_len = strlen(str);
    char *new_str = (char *)malloc(str_len + 1);
    memcpy(new_str, str, str_len);
    new_str[str_len] = c;
    return new_str;
}

char    *__internal_find_code(t_tree *tree, u_int8_t c, char *current) {
    char *ret;

    if (tree == NULL)
        return NULL;
    if (tree->c == c)
        return current;
    char *new_string = strdup(current);
    new_string = append(new_string, '0');
    ret = __internal_find_code(tree->left, c, new_string);
    if (ret == NULL) {
        new_string[strlen(new_string) - 1] = '1';
        ret = __internal_find_code(tree->right, c, new_string);
    }
//    free(new_string);
    return ret;
}

char *find_code(t_tree *tree, u_int8_t c)
{
    return __internal_find_code(tree, c, "");
}

int 	main(int argc, char **argv)
{
    t_symbol	**hash;
    t_symbol    **unsort_hash;
    u_int32_t	sym_cnt;

    if (argc == 1)
    {
        printf("no input files\n");
        return (1);
    }
    if (argc == 2)
    {
        printf("no output files\n");
        return (1);
    }
    hash = (t_symbol **) malloc (256 * sizeof(t_symbol *));
    unsort_hash = (t_symbol **) malloc (256 * sizeof(t_symbol *));
    for (u_int16_t i=0; i < 256; ++i)
    {
        hash[i] = (t_symbol *) malloc (sizeof(t_symbol));
        hash[i]->cnt = 0;
        hash[i]->sym = i;
        hash[i]->code = "";

        unsort_hash[i] = (t_symbol *) malloc (sizeof(t_symbol));
        unsort_hash[i]->cnt = 0;
        unsort_hash[i]->sym = i;
        unsort_hash[i]->code = "";
    }
    if (read_file(argv[1], hash, &sym_cnt))
    {
        printf("cannot open %s file\n", argv[1]);
        return (0);
    }
    for (u_int16_t i=0; i < 256; ++i)
        if (hash[i]->cnt != 0)
            printf("%4d (%c): %4d\n", hash[i]->sym, hash[i]->sym, hash[i]->cnt);
    for (u_int16_t i=0; i < 256; ++i)
        unsort_hash[i]->cnt = hash[i]->cnt;
    sort(hash);
    printf("\n");
    for (u_int16_t i=0; i < 256; ++i)
        if (hash[i]->cnt != 0)
            printf("%4d (%c): %4d\n", hash[i]->sym, hash[i]->sym, hash[i]->cnt);
    t_tree *tree = *archivate(hash);

    for (u_int16_t i=0; i < 256; ++i)
    {
        unsort_hash[i]->code = find_code(tree, unsort_hash[i]->sym);
        hash[i]->code = find_code(tree, hash[i]->sym);
    }

    int out = open(argv[2], (unsigned)O_CREAT | (unsigned)O_WRONLY);
    if (out == -1) {
        printf("cannot open/create output file\n");
        return (1);
    }
    printf("\n");
    for (u_int16_t i=0; i < 256; ++i) {
        if (hash[i]->cnt != 0) {
            printf("%4d (%c): \"%s\"\n", hash[i]->sym, hash[i]->sym, hash[i]->code);
            write(out, &hash[i]->sym, 1); // sym
            write(out, "=", 1);
            write(out, hash[i]->code, strlen(hash[i]->code));
        }
    }
    int fd = open(argv[1], O_RDONLY);
    char c = 0;
    write(out, &c, 1);
    while (read(fd, &c, 1) != 0) {
        write_next_char(out, unsort_hash[c]->code);
    }
    write_next_char(out, NULL);
    close(out);
    close(fd);
    return (0);
}

t_tree	*new_node(u_int32_t data, char c)
{
    t_tree *new_tree;

    new_tree = (t_tree *)malloc(sizeof(t_tree));
    if (new_tree == NULL)
        return new_tree;
    new_tree->data = data;
    new_tree->c = c;
    new_tree->left = NULL;
    new_tree->right = NULL;
    return new_tree;
}

void	tree_swap(t_tree **t1, t_tree **t2)
{
    t_tree	*ptr = *t1;
    *t1 = *t2;
    *t2 = ptr;
}

void     write_next_char(int fd, char *code) {
    static unsigned int wrote = 0;
    static unsigned char c = 0;

    if (code == NULL)
    {
        if (wrote != 0)
            write(fd, &c, 1);
        c = wrote + 48;
        write(fd, &c, 1);
        return;
    }
    while (*code) {
        c |= (unsigned char)(((unsigned char)(*code - 48)) << (7u - wrote));
        wrote++;
        if (wrote == 8) {
            write(fd, &c, 1);
            wrote = 0;
            c = 0;
        }
        code++;
    }
}

t_tree **archivate(t_symbol **hash)
{
    t_tree		**array;
    t_tree		*new_leaf;
    u_int16_t	size;

    size = 0;
    for (u_int16_t i=0; i < 256; ++i)
        if (hash[i]->cnt > 0)
            size++;
    array = (t_tree **)malloc(sizeof(t_tree *) * size);
    for (u_int16_t i=0; i < size; ++i)
        array[i] = new_node(hash[i]->cnt, hash[i]->sym);
    for (u_int16_t i=size - 1; i > 0; --i)
    {
        size--;
        new_leaf = new_node(array[size]->data + array[size - 1]->data, 0);
        new_leaf->left = array[size - 1];
        new_leaf->right = array[size];
        array[size - 1] = new_leaf;
        u_int16_t j = size - 1;
        while (j > 0 && array[j - 1]->data < array[j]->data)
        {
            tree_swap(&array[j - 1], &array[j]);
            --j;
        }
    }
    return array;
}