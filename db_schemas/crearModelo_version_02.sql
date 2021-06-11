CREATE DATABASE  IF NOT EXISTS `wifialr` /*!40100 DEFAULT CHARACTER SET utf8 */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `wifialr`;
-- MySQL dump 10.13  Distrib 8.0.21, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: wifialr
-- ------------------------------------------------------
-- Server version	8.0.21

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `alertas`
--

DROP TABLE IF EXISTS `alertas`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `alertas` (
  `ID` int NOT NULL AUTO_INCREMENT,
  `ALERTA` varchar(200) NOT NULL,
  `FK_MOVIMIENTO` int NOT NULL,
  PRIMARY KEY (`ID`),
  KEY `FK_MOVIMIENTO_idx` (`FK_MOVIMIENTO`),
  CONSTRAINT `FK_ALERTAS_MOVIMIENTOS` FOREIGN KEY (`FK_MOVIMIENTO`) REFERENCES `movimientos` (`ID`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `alertas`
--

LOCK TABLES `alertas` WRITE;
/*!40000 ALTER TABLE `alertas` DISABLE KEYS */;
INSERT INTO `alertas` VALUES (1,'HAS GONE TO BED',1),(2,'HAS FALLEN',2),(3,'HAS LIFTED AN OBJECT',3),(4,'HAS RUN',4),(5,'HAS SAT DOWN',5),(6,'HAS STOOD UP',6),(7,'HAS WALKED',7);
/*!40000 ALTER TABLE `alertas` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `min_max`
--

DROP TABLE IF EXISTS `min_max`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `min_max` (
  `ID` int NOT NULL AUTO_INCREMENT,
  `ANTENA_1_AMP_SUB1` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB2` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB3` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB4` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB5` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB6` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB7` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB8` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB9` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB10` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB11` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB12` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB13` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB14` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB15` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB16` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB17` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB18` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB19` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB20` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB21` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB22` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB23` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB24` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB25` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB26` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB27` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB28` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB29` decimal(18,15) NOT NULL,
  `ANTENA_1_AMP_SUB30` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB1` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB2` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB3` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB4` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB5` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB6` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB7` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB8` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB9` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB10` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB11` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB12` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB13` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB14` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB15` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB16` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB17` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB18` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB19` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB20` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB21` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB22` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB23` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB24` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB25` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB26` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB27` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB28` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB29` decimal(18,15) NOT NULL,
  `ANTENA_2_AMP_SUB30` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB1` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB2` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB3` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB4` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB5` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB6` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB7` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB8` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB9` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB10` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB11` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB12` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB13` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB14` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB15` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB16` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB17` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB18` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB19` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB20` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB21` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB22` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB23` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB24` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB25` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB26` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB27` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB28` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB29` decimal(18,15) NOT NULL,
  `ANTENA_3_AMP_SUB30` decimal(18,15) NOT NULL,
  `VALOR` varchar(200) NOT NULL COMMENT 'PUEDE SER "MINIMO" O  "MAXIMO"',
  `DATE_CREATED` datetime NOT NULL,
  `FK_MOVIMIENTO` int NOT NULL,
  `ASOCIADO` int DEFAULT NULL COMMENT 'CUANDO ES UN REGISTRO DE VALORES EN "MINIMO", SE DEJA VACÍO ESTE CAMPO; CUANDO ES UN REGISTRO DE VALORES EN "MAXIMO", ESTE CAMPO TENDRÁ EL VALOR DEL ID DE SU VALOR "MINIMO" ASOCIADO.\n.',
  PRIMARY KEY (`ID`),
  KEY `FK_MOVIMIENTO_idx` (`FK_MOVIMIENTO`),
  CONSTRAINT `FK_MIN_MAX_MOVIMIENTO` FOREIGN KEY (`FK_MOVIMIENTO`) REFERENCES `movimientos` (`ID`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `min_max`
--

LOCK TABLES `min_max` WRITE;
/*!40000 ALTER TABLE `min_max` DISABLE KEYS */;
/*!40000 ALTER TABLE `min_max` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `movimientos`
--

DROP TABLE IF EXISTS `movimientos`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `movimientos` (
  `ID` int NOT NULL AUTO_INCREMENT,
  `MOVIMIENTO` varchar(200) NOT NULL,
  `SIGNIFICADO` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `movimientos`
--

LOCK TABLES `movimientos` WRITE;
/*!40000 ALTER TABLE `movimientos` DISABLE KEYS */;
INSERT INTO `movimientos` VALUES (1,'BE','GO TO BED'),(2,'FA','FALL'),(3,'PI','PICK UP'),(4,'RU','RUN'),(5,'SD','SIT DOWN'),(6,'SU','STAND UP'),(7,'WA','WALK'),(8,'NA','DOES NOT APPLY');
/*!40000 ALTER TABLE `movimientos` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-10-20  1:30:32
