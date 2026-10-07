# Scoring evidence

Source: [doppelganger SMB disassembly mirror](https://github.com/MrWint/smb-dis/blob/master/smbdis.asm); downloaded source SHA-256 is in the JSON fixture. No ROM or assembly is included. Wrapper: local gym-super-mario-bros 9.1.0, smb_env.py.

| Evidence | Interpretation |
| --- | --- |
| CheckForCoinMTiles | 0xc2 / 0xc3 identify coin metatiles. |
| BrickQBlockMetatiles / BumpBlock | 0xc0 dispatches to CoinBlock as a question block; 0x58 / 0x5d dispatch to CoinBlock as multi-coin bricks. 0xc1 is a mushroom/flower block, not a coin block. |
| PlayerHeadCollision / BlockBumpedChk | Bumping a coin block replaces its buffer tile temporarily with 0x23; the stored replacement is 0xc4 when consumed, or the original multi-coin tile while its timer permits. An observed coin brick does not reveal an exact remaining count. |
| GiveOneCoin | Coin count wraps at 100; an extra life accompanies rollover. |
| HandleStompedShellE | Goomba / walking Koopa transitions to state 4; Mario vertical speed becomes 0xfc. |
| PlayerEnemyCollision | Collision bit 0 corroborates contact; star defeats follow another path. |
| ForceInjury | Powered Mario becomes small, engine state 0x0a and injury timer is set. |
| FloateyNumbersRoutine | Enemy score is applied later than contact. |
| Wrapper _score / _coins | info reads decimal digits at 0x07de / 0x07ed. |
| ScreenLeft / ScreenRight | World bounds use page bytes 0x071a/0x071b and low bytes 0x071c/0x071d. |
| BlockBufferAddr | Two 208-byte pages start at 0x0500; columns wrap by world page parity; rows start at screen y=32. |

`stomp` is a recorded Goomba contact. At frame 121 Mario bottom=192, enemy top=196; at 122 bottom=197, enemy top=196, collision bit becomes 1, enemy state becomes 4 and Mario vertical speed becomes 252 (-4). Score remains 0: a stomp is not the same event as a score change.

`coin` is a recorded counter increment at frame 998, score 100→300 and coins 0→1. It confirms a collected coin, not the identity of a disappearing visible tile (block coins are also possible).

The other cases are explicitly synthetic. Additional synthetic negative cases for side contact, slot reuse, ambiguous simultaneous gains, upgrade vs injury, short RAM, and missing telemetry are constructed in test_scoring.py. No recorded rollover, powered injury or Koopa stomp was found in the sampled local traces; rule-based support must remain conservative and coverage limitations must be reported. Only ordinary walking Goomba/green Koopa states 0/1 in non-water areas qualify as potential stomp targets; actual event confirmation requires additional contact evidence. Shell kicking is excluded.

All expected events are independently checked against the source and frame geometry. Original trace filenames and episode numbers identify recorded samples; full logs remain ignored. Counter discontinuities and object disappearance alone are not positive events.

`coin-block-observations.json` contains two unmodified RAM/info frames from the
reported gameplay trace `run-20261007T101101.875745Z.trace.jsonl`:

- Episode 0, frame 89, Mario x=240: coin question tiles at world (256,144),
  (352,80), (368,144); the tile at (336,144) is 0xc1 and must not be a coin target.
- Episode 1, frame 802, Mario x=1409: a coin question tile at (1504,80) and
  multi-coin brick 0x58 at (1504,144).

Before the fix, all five coin-block observations were absent from model input.
Regression tests also cover synthetic used/bouncing blocks, power-up/hidden tiles,
page-buffer parity, viewport exclusion and preserving a block in Mario's current
column. Hidden coin tile 0x5f is intentionally excluded from visible targets.
These verify observations only, not autonomous block-targeting success.

For block-targeting geometry, `PlayerBGCollision`, `BlockBufferAdderData` and
`BlockBuffer_X_Adder` / `BlockBuffer_Y_Adder` select a background head probe at
world x+8 and screen y+18 for small/crouching Mario, or y+4 for standing big Mario
outside swimming. This differs from the object collision-box top. The relative
horizontal interval uses [left-head_x, right-head_x); the required rise is
head_y minus block bottom, so a positive value means the head is below the block.
`BlockBounceTimer` at 0x0784 inhibits another block bump while nonzero. These
fields describe current geometry, not future collision success or a takeoff rule.

Power-up support follows `SetupPowerUp`: metatile 0xc1 calls `MushFlowerBlock`,
using `PlayerStatus` (0x0756) at the hit to choose mushroom (0) or flower (1).
Slot 5 uses active 0x0014, id 0x001b=0x2e, state 0x0023 and type 0x0039;
world X is 0x0073/0x008c, Y is 0x00bb/0x00d4. `PowerUpObjHandler` skips
collision-box refresh before state 6, so early emergence never exposes stale
box data. Later bounds use 0x04c4..0x04c7 and masked offscreen bits 0x03dd.
`MoveNormalEnemy` / `Chk2MSBSt` permit mushroom states 0x80 (moving) and
0xc0 (falling); `NMovShellFallBit` clears the fall bit on landing. Signed Q4
speed is at 0x005d. Flowers and emerging objects have no horizontal movement.
An emergence-state regression identifies a replacement in the reused slot even
without an intervening inactive frame. `HandlePowerUpCollision` grants actual
upgrades (small to big or big to fiery); a spawned or disappeared object alone
does not establish collection. Synthetic tests cover both types, emergence,
falling, slot replacement, viewport/bbox exclusions and policy prompt activation.

`powerup-emergence.json` is an unmodified frame from a development full run:
frame 768 after a 0xc1 head hit at frame 762, with a mushroom at world x=1248.
The object is still emerging, so its collision box must be null in model input.
The fixture verifies observation of an actual spawned object, not successful
collection or improved gameplay. The baseline parser fails this regression case.
