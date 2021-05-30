UPDATE `wifialr`.`movimientos` SET `MOVIMIENTO` = 'SD', `SIGNIFICADO` = 'SIT DOWN' WHERE (`ID` = '5');
UPDATE `wifialr`.`movimientos` SET `MOVIMIENTO` = 'SU', `SIGNIFICADO` = 'STAND UP' WHERE (`ID` = '6');

UPDATE `wifialr`.`alertas` SET `ALERTA` = 'SE HA SENTADO' WHERE (`ID` = '5');
UPDATE `wifialr`.`alertas` SET `ALERTA` = 'SE HA PARADO' WHERE (`ID` = '6');