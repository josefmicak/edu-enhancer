# EduEnhancer
Aplikace sloužící k práci s testy (přidávání a vyplňování testů). Aplikace byla vytvořena za využití platformy ASP.NET Core s .NET 6 pomocí programovacího jazyka C#.

## Seznam funkcí
- přidávání a úprava testů učiteli nebo správci
- vyplňování testů studenty
- podpora 10 různých typů otázek
- bodovací systém poskytující různé možnosti, jako např. nastavení přenášení záporných bodů mezi otázkami pro každý test nebo nastavení počtu odečtených bodů za špatnou volbu
- meziplatformnost (snadný přechod mezi operačními systémy Windows a Ubuntu)
- přihlašovací systém používající Google účty k autentizaci uživatele
- uživatelský systém s různými rolemi a právy (hlavní správce/správce/učitel/student)
- využití umělé inteligence pro různé úkony (doporučený počet bodů za zadání otázky, doporučený počet bodů za studentovu odpověď za otázku, odhadovaná obtížnost daného testu v porování s ostatními testy)

## O projektu
Jedná se o projekt volně navazující na projekt pojmenovaný "TaoEnhancer", který byl nejprve vytvářen v týmu dvou studentů v rámci semestrálního projektu na Katedře informatiky VŠB-TUO FEI. V rámci TaoEnhanceru se počítalo s podporou nástroje TAO Core pomocí kterého byly prováděny určité funkce (jako např. vytváření a vyplňovaní testů) v současné době je projekt (nyní již jedním studentem) rožšiřován v rámci diplomového projektu na stejné katedře a je zcela nezávislý na nástroji TAO Core (veškeré funkce související s TAO Core byly z projektu odebrány). Nejvýznamnějším aspektem projektu je využití umělé inteligence, která je zde využívána pro určité funkce, a pro jejíž implementaci jsou využity neuronové sítě a strojové učení.