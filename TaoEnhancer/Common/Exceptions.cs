using System.IO;
using System.Xml;

namespace Common
{
    /// <summary>
    /// A class containing all user-made exceptions
    /// </summary>
    public static class Exceptions
    {
        //Filesystem exceptions
        public static DirectoryNotFoundException TestTemplatesPathNotFoundException { get { return new DirectoryNotFoundException("Chyba: složka s šablonami testů nebyla nalezena."); } }
        public static DirectoryNotFoundException QuestionTemplatesPathNotFoundException(string testNameIdentifier)
        {
            return new DirectoryNotFoundException("Chyba: složka s šablonami otázek nebyla nalezena. Identifikátor testu: " + testNameIdentifier);
        }
        public static DirectoryNotFoundException TestResultsPathNotFoundException { get { return new DirectoryNotFoundException("Chyba: složka s výsledky testů nebyla nalezena."); } }
        public static DirectoryNotFoundException StudentsPathNotFoundException { get { return new DirectoryNotFoundException("Chyba: složka se studenty nebyla nalezena."); } }
        public static FileNotFoundException TestTemplateNotFoundException(string testResultIdentifier)
        {
            return new FileNotFoundException("Chyba: šablona testu nebyla nalezena. Identifikátor testu: " + testResultIdentifier);
        }
        public static FileNotFoundException QuestionTemplateNotFoundException(string testResultIdentifier, string questionNumberIdentifier)
        {
            return new FileNotFoundException("Chyba: šablona otázky nebyla nalezena. Identifikátor testu: " + testResultIdentifier + ", identifikátor otázky: " + questionNumberIdentifier);
        }

        //XML exceptions
        public static XmlException StudentsAnswerNotFoundException(string testNameIdentifier, string questionNumberIdentifier, string subquestionIdentifier)
        {
            return new XmlException("Chyba: studentova odpověď nebyla nalezena (identifikátor testu: " + testNameIdentifier +
                ", identifikátor otázky: " + questionNumberIdentifier + ", identifikátor podotázky: " + subquestionIdentifier + ")");
        }

        //Data exceptions
        public static Exception SubquestionTemplateNotFoundException { get { return new Exception("Chyba: šablona podotázky nebyla nalezena."); } }
        public static Exception SubquestionResultNotFoundException { get { return new Exception("Chyba: výsledek podotázky nebyl nalezen."); } }
        public static Exception UserNotFoundException { get { return new Exception("Chyba: uživatel nebyl nalezen."); } }
        public static Exception SpecificUserNotFoundException(string login)
        {
            return new Exception("Chyba: uživatel s loginem " + login + " nebyl nalezen.");
        }
        public static Exception UserAlreadyExistsException(string login)
        {
            return new Exception("Chyba: uživatel s loginem " + login + " již existuje.");
        }
        public static Exception StudentNotFoundException(string studentIdentifier)
        {
            return new Exception("Chyba: student s identifikátorem " + studentIdentifier + " nebyl nalezen.");
        }
        public static Exception StudentsNotImportedException { get { return new Exception("Chyba: do systému nebyli zatím importováni žádní studenti."); } }
        public static Exception TestTemplatesNotImportedException { get { return new Exception("Chyba: do systému nebyli zatím importovány žádné šablony testu."); } }
        public static Exception AccessDeniedException { get { return new Exception("Chyba: přístup zamítnut."); } }
        public static Exception NoElementsFoundException { get { return new Exception("Chyba: stránku nelze zobrazit."); } }
    }
}
