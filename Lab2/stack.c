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
#define MAX_CHUNK 5000


static stack_t free_list_array[NB_THREADS];
static stack_t* chunks_to_free[MAX_CHUNK];
static nb_to_free = 0;

void init_free_list_array() {
	for (int i=0; i < NB_THREADS; ++i ){
		free_list_array[i].head = malloc(sizeof(cell_t) * CHUNK_SIZE);
		// link all the cells in the chunk
		for (int j=0; j < CHUNK_SIZE; ++j) {
			((cell_t *) free_list_array[i].head + j )->next = (cell_t *) free_list_array[i].head + j+1;
		}
		((cell_t *) (free_list_array[i].head + CHUNK_SIZE - 1))->next = NULL;

	}
}

void free_free_list_array(){
	// TODO - FIX LEAKS + UNIT TESTS
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
	assert(s != NULL);
	assert(thread_id >= 0);


	// if free_list is empty (ie all cells are used), reallocate CHUNK
	if(free_list_array[thread_id].head == NULL){
		// push the chunk to free_chunks
		chunks_to_free[nb_to_free++] = &free_list_array[thread_id];

		// allocate new chunk
		free_list_array[thread_id].head = malloc(sizeof(cell_t) * CHUNK_SIZE);
		
		// link all the cells
		for (int j=0; j + 1 < CHUNK_SIZE; ++j) {
			((cell_t *) (free_list_array[thread_id].head + j))->next = ((cell_t *) (free_list_array[thread_id].head + (j+1)));
		}
		((cell_t *) (free_list_array[thread_id].head + CHUNK_SIZE - 1))->next = NULL;


	} // end if

	stack_t* free_list = &free_list_array[thread_id];
	cell_t* c;
	cell_t* tmp;
	// pops from free list
	// old = s->head;
	// c->next = old;
	tmp = free_list->head;
	free_list->head = free_list->head->next;
	c = tmp;
	c->val = val;

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

#endif


  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
  return 0;
}

int stack_pop(stack_t* s, int thread_id) {
  assert(s != NULL);
	// if (s->head == NULL){
	// 	printf("%free_list->head : %p\n", free_list_array[thread_id].head );
	// 	printf("%free_list->index : %d\n", free_list_array[thread_id].index );
	// 	printf("%free_list->next : %p\n", free_list_array[thread_id].next_chunk );
	//
	// }
	assert(s->head != NULL);
	assert(thread_id >= 0);

	cell_t* old;
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&mtx);
	old = s->head;
  s->head = s->head->next;
	// printf("popped : %d by t%d\n",old->val,thread_id);
  pthread_mutex_unlock(&mtx);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
	cell_t* newHead;
	do {
			old = s->head;
			newHead = s->head->next;
	} while(cas(((size_t*)&(s->head)), ((size_t)(old)), ((size_t)newHead)) != (size_t)old);
	#endif

	// pop from shared stack means pushes it to the free_list of the thread
	// c->next = s->head;
	// s->head = c;
	old->next = free_list_array[thread_id].head;
	free_list_array[thread_id].head = old;



  return 0;

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
