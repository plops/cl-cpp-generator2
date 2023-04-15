- Aztec monks with 1/55 HP no longer die when picking up or dropping a
  relic (reddit.com)
- https://news.ycombinator.com/item?id=35539595
- https://old.reddit.com/r/Games/comments/12jbb9d/age_of_empires_ii_de_major_update_81058/


> The formula was probably `return new_maxhp * (old_hp / old_maxhp);`
> ...
> Due to floating point rounding errors, convert_hp(1, 55, 55) equals
> 0.999999940395355224609375, which is less than 1, which means the
> unit dies.


- std::ratio is only for compile time. there doesn't seem to be
  standard rational type for c++ (except by pulling in boost)
