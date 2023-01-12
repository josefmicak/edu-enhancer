using System.IO;
using System.Xml;

namespace Common
{
    /// <summary>
    /// A class containing all user-made exceptions
    /// </summary>
    public static class Exceptions
    {
        //Data exceptions
        public static Exception SubquestionTemplateNotFoundException { get { return new Exception("Chyba: šablona podotázky nebyla nalezena."); } }
        public static Exception SubquestionResultNotFoundException { get { return new Exception("Chyba: výsledek podotázky nebyl nalezen."); } }
        public static Exception UserNotFoundException { get { return new Exception("Chyba: uživatel nebyl nalezen."); } }
        public static Exception UserAlreadyExistsException(string login)
        {
            return new Exception("Chyba: uživatel s loginem " + login + " již existuje.");
        }
        public static Exception AccessDeniedException { get { return new Exception("Chyba: přístup zamítnut."); } }
        public static Exception NoElementsFoundException { get { return new Exception("Chyba: stránku nelze zobrazit."); } }
    }
}
