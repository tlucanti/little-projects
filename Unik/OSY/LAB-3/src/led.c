/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   led.c                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: kostya <kostya@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/11/29 17:17:41 by kostya            #+#    #+#             */
/*   Updated: 2021/11/29 19:24:14 by kostya           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

# include "../inc/lab3.h"

static int __internal_led_visor(int action) __NOEXC;

__NOEXC
void swap_led(void)
{
    printf("\tswap led (now %d)\n", __internal_led_visor('s'));
}

__NOEXC
void on_led(void)
{
    printf("\tturned led on (now %d)\n", __internal_led_visor('1'));
}

__NOEXC
void off_led(void)
{
    printf("\tturned led off (now %d)\n", __internal_led_visor('0'));
}

__NOEXC
static int __internal_led_visor(int action)
{
    static int _led = 0;

    if (action == '0')
        _led = 0;
    else if (action == '1')
        _led = 1;
    else if (action == 's')
        _led ^= 1;
    return _led;
}
