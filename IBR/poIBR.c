
#include <stdint.h>
#include <stddef.h>
#include <malloc.h>
#include <stdlib.h>

#define PACKED	__attribute__((__packed__))
#define INF	(uint32_t)(-1)
typedef int	bool;
#define true	1
#define false	0

void atomic_fetch_add(void *, size_t);
bool atomic_compare_exchange(void *, void *, void *);


// ----------------------------------------------------------------------------
typedef union poibr_interval_u
{
	uint64_t	value;
	struct		PACKED { uint32_t birth_epoch; uint32_t reclaim_epoch; };
} PACKED poibr_interval_t;

// ----------------------------------------------------------------------------
typedef struct poibr_block_s
{
	poibr_interval_t	life_time;
	void			*pointer;
} PACKED poibr_block_t;

// ----------------------------------------------------------------------------
#define POIBR_THREAD_COUNT	20
#define POIBR_RECLAIM_COUNT	100
#define POIBR_EPOCH_FREQ	5
#define POIBR_CLEAR_FREQ	50

uint32_t	g_epoch;
uint32_t	reservations[POIBR_THREAD_COUNT];
uint32_t	alloc_counter[POIBR_THREAD_COUNT];
poibr_block_t	*wait_list[POIBR_THREAD_COUNT][POIBR_RECLAIM_COUNT];
uint32_t	wait_list_counter[POIBR_THREAD_COUNT];
uint32_t	reclaim_counter[POIBR_THREAD_COUNT];

// ============================================================================
// ----------------------------------------------------------------------------
void poibr_enter(int tid)
{
	reservations[tid] = g_epoch;
}

// ----------------------------------------------------------------------------
void poibr_leave(int tid)
{
	reservations[tid] = INF;
}

// ----------------------------------------------------------------------------
poibr_block_t	*poibr_allocate(int tid, size_t size)
{
	void		*ptr;
	poibr_block_t	*block;

	if (++alloc_counter[tid] % POIBR_EPOCH_FREQ == 0)
		atomic_fetch_add(&g_epoch, 1);
	ptr = malloc(size);
	block = (poibr_block_t *)malloc(sizeof(poibr_block_t));
	if (ptr == NULL || block == NULL)
		abort();
	block->pointer = ptr;
	block->life_time.birth_epoch = g_epoch;
	return block;
}

// ----------------------------------------------------------------------------
void	_poibr_clear(int tid)
{
	poibr_block_t		**w_l = wait_list[tid];
	bool			conflict;
	poibr_interval_t	life;
	uint32_t		res;
	uint32_t		place_i = 0;

	for (uint32_t block_i=0; block_i < wait_list_counter[tid]; ++block_i)
	{
		conflict = false;
		for (uint32_t res_i = 0; res_i < POIBR_THREAD_COUNT; ++res_i)
		{
			life = w_l[block_i]->life_time;
			res = reservations[res_i];
			if (life.birth_epoch <= res
				&& res <= life.reclaim_epoch)
			{
				conflict = true;
				break ;
			}
		}
		if (!conflict)
		{
			free(w_l[block_i]->pointer);
			free(w_l[block_i]);
		}
		else
		{
			w_l[place_i] = w_l[block_i];
			++place_i;
		}
	}
	wait_list_counter[tid] = place_i;
}

// ----------------------------------------------------------------------------
void	poibr_reclaim(int tid, poibr_block_t *block)
{
	wait_list[tid][ ++wait_list_counter[tid] ] = block;
	block->life_time.reclaim_epoch = g_epoch;
	if (++reclaim_counter[tid] % POIBR_CLEAR_FREQ == 0)
		_poibr_clear(tid);
}

// ----------------------------------------------------------------------------
poibr_block_t	*poibr_read(int tid, poibr_block_t **ptr)
{
	poibr_block_t	*ret;

	do
	{
		reservations[tid] = g_epoch;
		ret = *ptr;
	} while (reservations[tid] != g_epoch);
	return ret;
}

// ----------------------------------------------------------------------------
void	poibr_write(poibr_block_t **ptr, poibr_block_t *block)
{
	*ptr = block;
	/* maybe use here atomic_write
	 */
}

// ----------------------------------------------------------------------------
bool	poibr_compare_exchange(
		poibr_block_t **ptr,
		poibr_block_t *old,
		poibr_block_t *new
	)
{
	return atomic_compare_exchange(ptr, old, new);
}

