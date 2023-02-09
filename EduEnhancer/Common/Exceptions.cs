using System.IO;
using System.Xml;

namespace Common
{
    /// <summary>
    /// A class containing all user-made exceptions
    /// </summary>
    public static class Exceptions
    {
        public static Exception SubquestionTemplateNotFoundException { get { return new Exception("Chyba: šablona podotázky nebyla nalezena."); } }
        public static Exception SubquestionResultNotFoundException { get { return new Exception("Chyba: výsledek podotázky nebyl nalezen."); } }
        public static Exception InvalidSubquestionResultIndexException { get { return new Exception("Chyba: neplatný index podotázky."); } }
        public static Exception UserLoggedOutException()
        {
            return new Exception("Chyba: byl jste odhlášen (po 20 minutách neaktivity dojde k odhlášení).");
        }
        public static Exception UserNotFoundException(string login)
        {
            return new Exception("Chyba: uživatel s loginem " + login + " nebyl nalezen.");
        }
        public static Exception UserEmailNotFoundException(string login)
        {
            return new Exception("Chyba: uživatel s emailem " + login + " nebyl nalezen.");
        }
        public static Exception StudentNotFoundException(string login)
        {
            return new Exception("Chyba: student s loginem " + login + " nebyl nalezen.");
        }
        public static Exception SubjectNotFoundException(int subjectId)
        {
            return new Exception("Chyba: předmět s Id " + subjectId + " nebyl nalezen.");
        }
        public static Exception SubquestionTemplateStatisticsNotFoundException(string login)
        {
            return new Exception("Chyba: statistiky zadání podotázek uživatele s loginem " + login + " nebyly nalezeny.");
        }
        public static Exception SubquestionResultStatisticsNotFoundException(string login)
        {
            return new Exception("Chyba: statistiky výsledků podotázek uživatele s loginem " + login + " nebyly nalezeny.");
        }
        public static Exception TestDifficultyStatisticsNotFoundException(string login)
        {
            return new Exception("Chyba: statistiky obtížnosti testů uživatele s loginem " + login + " nebyly nalezeny.");
        }
        public static Exception AccessDeniedException { get { return new Exception("Chyba: přístup zamítnut."); } }
        public static Exception NoElementsFoundException { get { return new Exception("Chyba: stránku nelze zobrazit."); } }
        public static Exception PythonException(string fileName, string functionName, string stderr)
        {
            return new Exception("Chyba: při práci s python skriptem došlo k chybě." +
                "\nJméno souboru: " + fileName + 
                "\nJméno funkce: " + functionName + 
                "\nChybová hláška: " + stderr);
        }
        public static Exception GlobalSettingsNotFound { get { return new Exception("Chyba: globální nastavení nenalezena."); } }
    }
}
