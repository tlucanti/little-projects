
#include <stdint.h>
#include <stddef.h>
#include <malloc.h>
#include <stdlib.h>

#define TWO_GLOBAL_EPOCHS_IBR 1

#define PACKED	__attribute__((__packed__))
#define INF	(uint64_t)(-1)
typedef int	bool;
#define true	1;
#define false	0;

void atomic_fetch_add(void *, size_t);
void atomic_store(void *, size_t);
bool atomic_compare_exchange(void *, void *, void *);
int max(uint32_t, uint32_t);

// ----------------------------------------------------------------------------
typedef union tibr_interval_u
{
	uint64_t	value;
	struct		PACKED { uint32_t lower; uint32_t upper; };
	struct		PACKED { uint32_t birth_epoch; uint32_t reclaim_epoch; };
} tibr_interval_t;

// ----------------------------------------------------------------------------
typedef struct tibr_pointer_s
{
	tibr_interval_t		life_time;
	void			*pointer;
} PACKED tibr_block_t;

// ----------------------------------------------------------------------------
typedef struct tibr_tagged_pointer_s
{
	tibr_block_t		*block;
	uint64_t		last_access;
} PACKED tibr_tagged_pointer_t;

// ----------------------------------------------------------------------------
#define TIBR_THREAD_COUNT	20
#define TIBR_RECLAIM_COUNT	100
#define TIBR_EPOCH_FREQ		5
#define TIBR_CLEAR_FREQ		50

uint32_t		g_epoch;
tibr_interval_t		reservations[TIBR_THREAD_COUNT];
uint32_t		alloc_counter[TIBR_THREAD_COUNT];
tibr_block_t		*wait_list[TIBR_THREAD_COUNT][TIBR_RECLAIM_COUNT];
uint32_t		wait_list_counter[TIBR_THREAD_COUNT];
uint32_t		reclaim_counter[TIBR_THREAD_COUNT];

// ============================================================================
// ----------------------------------------------------------------------------
void	tibr_enter(int tid)
{
	reservations[tid].lower = g_epoch;
	reservations[tid].upper = g_epoch;
}

// ----------------------------------------------------------------------------
void	tibr_leave(int tid)
{
	reservations[tid].value = INF;
}

// ----------------------------------------------------------------------------
tibr_block_t	*tibr_allocate(int tid, size_t size)
/* maybe here we will get pointer with data and set it to tibr_block_t->pointer
 * instead of allocating it inside this function (because otherwise we will
 * need to call tibr_write right after tibr_allocate just to set this pointer
 */
{
	tibr_block_t	*block;
	void		*ptr;

	if (++alloc_counter[tid] % TIBR_EPOCH_FREQ == 0)
		atomic_fetch_add(&g_epoch, 1);
	ptr = malloc(size);
	block = (tibr_block_t *)malloc(sizeof(tibr_block_t));
	if (ptr == NULL || block == NULL)
		abort();
	block->pointer = ptr;
	block->life_time.birth_epoch = g_epoch;
	return block;
}

// ----------------------------------------------------------------------------
void	_tibr_clear(int tid)
{
	tibr_block_t		**w_l = wait_list[tid];
	bool			conflict;
	tibr_interval_t		life;
	tibr_interval_t		res;
	uint32_t		place_i = 0;

	for (uint32_t block_i=0; block_i < wait_list_counter[tid]; ++block_i)
	{
		conflict = false;
		for (uint32_t res_i=0; res_i < TIBR_THREAD_COUNT; ++res_i)
		{
			life = w_l[block_i]->life_time;
			res = reservations[res_i];
			if (life.birth_epoch <= res.upper
				&& life.reclaim_epoch >= res.lower)
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
void	recalim(int tid, tibr_block_t *block)
{
	wait_list[tid][ ++wait_list_counter[tid] ] = block;
	block->life_time.reclaim_epoch = g_epoch;
	if (++reclaim_counter[tid] % TIBR_CLEAR_FREQ == 0)
		_tibr_clear(tid);
	/* here if we will leave after TIBR_CLEAR_FREQ - 1 reclaims -
	 * _tibr_clear() will never called for current thread, so we will get
	 * memory leak and floating epochs witch will never freed
	 */
}

#if TWO_GLOBAL_EPOCHS_IBR == 1
// ----------------------------------------------------------------------------
tibr_block_t	*read(int tid, tibr_tagged_pointer_t *ptr)
{
	tibr_block_t	*ret;

	do
	{
		reservations[tid].upper = max(
			reservations[tid].upper, g_epoch);
		/* maybe we can here dereference freed memory if any other
		 * thread in this position will reclaim this block  and
		 * instantly clear
		 */
		ret = ptr->block;
	} while (reservations[tid].upper != g_epoch);
	return ret;
}

#else /* not two global epochs */
// ----------------------------------------------------------------------------
tibr_block_t	*read(int tid, tibr_tagged_pointer_t *ptr)
{
	tibr_block_t	*ret;

	do
	{
		reservations[tid].upper = max(
			reservations[tid].upper, ptr->last_access);
		ret = ptr->block;
	} while (reservations[tid].upper < ptr->last_access);
	return ret;
}
#endif /* two global epochs */

// ----------------------------------------------------------------------------
void	tibr_write(tibr_tagged_pointer_t *ptr, tibr_block_t *block)
{
	if (block->life_time.birth_epoch > ptr->last_access)
		atomic_store(&ptr->last_access, block->life_time.birth_epoch);
	ptr->block = block;
	/* maybe here we need to use atomic_store for ptr->block and check
	 * in loop if ptr->last_access not changed during atommic_store
	 */
}

// ----------------------------------------------------------------------------
bool	tibr_compare_exchange(
		tibr_tagged_pointer_t *ptr,
		tibr_block_t *old,
		tibr_block_t *new
	)
{

	if (new->life_time.birth_epoch > ptr->last_access)
		atomic_store(&ptr->last_access, new->life_time.birth_epoch);
	return atomic_compare_exchange(ptr->block, old, new);
	/* here is the same question with loop trying as in tibr_write
	 */
}

