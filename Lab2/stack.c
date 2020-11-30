/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

#define CHUNK_SIZE 1000
static stack_t* pool_array[NB_THREADS];

void init_pool() {
	for (int i=0; i < NB_THREADS; ++i ){
		pool_array[i] = malloc(sizeof(stack_t));
		pool_array[i]->head = malloc(sizeof(cell_t) * CHUNK_SIZE);
		pool_array[i]->index = 0;
		pool_array[i]->next_chunk = NULL;
	}
}

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass
	assert(1 == 1);
	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int stack_push(stack_t * s, int val, int thread_id)
{
	assert(s != NULL && thread_id >= 0);

	stack_t* pool = pool_array[thread_id];

	// if pool has no more element, reallocate CHUNK
	// c->next = s->head;
  // s->head = c;

	if(pool->index == CHUNK_SIZE){
		stack_t *new_chunk = malloc(sizeof(stack_t));
		new_chunk->head = malloc(sizeof(cell_t) * CHUNK_SIZE);
		new_chunk->index = 0;
		new_chunk->next_chunk = pool;
		pool = new_chunk;
	} // end while

	cell_t* c;
	cell_t* old;

	//p + i = adresse contenue dans p + i*taille(élément pointé par p)
	c = pool->head + pool->index;
	c->val = val;
	pool->index++;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
	pthread_mutex_lock(&mtx);
	c->next = s->head;
  s->head = c;
  pthread_mutex_unlock(&mtx);


#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
		do {
			old = s->head;
			c->next = old;
		} while(cas(((size_t*)&(s->head)), ((size_t)(old)), ((size_t)c)) != (size_t)old);

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
	#endif
  return 0;
}

cell_t* stack_pop(stack_t* s, int thread_id) {
  assert(s != NULL && thread_id >= 0); //&& !(pool_array[thread_id].index == 0 & pool_array[thread_id].next_chunk == NULL));
	cell_t* old;
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&mtx);
	old = s->head;
  s->head = s->head->next;
  pthread_mutex_unlock(&mtx);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
	cell_t* newHead;
	do {
			old = s->head;
			newHead = s->head->next;
	} while(cas(((size_t*)&(s->head)), ((size_t)(old)), ((size_t)newHead)) != (size_t)old);
	#endif

	stack_t* pool = pool_array[thread_id];
	// pop head means giving it back to the pool
	// old = s->head;
	// s->head = s->head->next;
	// free(old)

	if (pool->index == 0){
		stack_t* old_chunk = pool;
		pool = pool->next_chunk;
		free(old_chunk->head);
	}

	old->val = 0;
	old->next = NULL;
	pool->index--;


  return old;

}

void stack_print (stack_t * s) {
    cell_t* cur = s->head;
    printf("Stack : [");
    while (cur != NULL) {
        printf("%d, ", cur->val);
        cur = cur->next;
    }
    printf("] \n");
}
