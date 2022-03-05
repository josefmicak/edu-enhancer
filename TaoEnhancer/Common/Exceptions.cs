namespace Common
{
    public static class Exceptions
    {
        public static DirectoryNotFoundException PathNotFoundException { get { return new DirectoryNotFoundException("Kořenová složka nebyla nalezena!"); } }
        public static DirectoryNotFoundException ResultsPathNotFoundException { get { return new DirectoryNotFoundException("Složka s výsledky testů nebyla nalezena!"); } }
        public static DirectoryNotFoundException ResultPathNotFoundException { get { return new DirectoryNotFoundException("Složka s výsledky testu nebyla nalezena!"); } }
        public static FileNotFoundException ResultFilePathNotFoundException { get { return new FileNotFoundException("XML soubor s výsledkem testu nebyl nalezen!"); } }
        public static FileNotFoundException ResultResultsDataPathNotFoundException { get { return new FileNotFoundException("Datový soubor s výsledkem testu nebyl nalezen!"); } }
        public static DirectoryNotFoundException StudentsPathNotFoundException { get { return new DirectoryNotFoundException("Složka se studenty nebyla nalezena!"); } }
        public static FileNotFoundException StudentFilePathNotFoundException { get { return new FileNotFoundException("RDF soubor se studentem nebyl nalezen!"); } }
        public static DirectoryNotFoundException TestsPathNotFoundException { get { return new DirectoryNotFoundException("Složka s testy nebyla nalezena!"); } }
        public static DirectoryNotFoundException TestPathNotFoundException { get { return new DirectoryNotFoundException("Složka s testem nebyla nalezena!"); } }
        public static DirectoryNotFoundException TestItemsPathNotFoundException { get { return new DirectoryNotFoundException("Složka s otázkami z testu nebyla nalezena!"); } }
        public static DirectoryNotFoundException TestItemPathNotFoundException { get { return new DirectoryNotFoundException("Složka s otázkou z testu nebyla nalezena!"); } }
        public static FileNotFoundException TestItemFilePathNotFoundException { get { return new FileNotFoundException("XML soubor s otázkou z testu nebyl nalezen!"); } }
        public static FileNotFoundException TestItemPointsDataPathNotFoundException { get { return new FileNotFoundException("Datový soubor s body za otázku z testu nebyl nalezen!"); } }
        public static FileNotFoundException TestItemImagePathNotFoundException { get { return new FileNotFoundException("Obrázek k otázce z testu nebyl nalezen!"); } }
        public static DirectoryNotFoundException TestTestsPathNotFoundException { get { return new DirectoryNotFoundException("Složka s konkrétními testy nebyla nalezena!"); } }
        public static DirectoryNotFoundException TestTestPathNotFoundException { get { return new DirectoryNotFoundException("Složka s konkrétním testem nebyla nalezena!"); } }
        public static FileNotFoundException TestTestFilePathNotFoundException { get { return new FileNotFoundException("XML soubor s konkrétním testem nebyl nalezen!"); } }
        public static FileNotFoundException TestTestNegativePointsDataPathNotFoundException { get { return new FileNotFoundException("Datový soubor se zápornými body konkrétního testu nebyl nalezen!"); } }
    }
}