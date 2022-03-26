namespace ViewLayer.Models
{
    public class ManageSolvedTestListModel : PageModel
    {
        private List<(string, string, string, string, string, string, string, string)> solvedTests = new List<(string, string, string, string, string, string, string, string)>();
        
        public List<(string, string, string, string, string, string, string, string)> SolvedTests { get { return solvedTests; } set { solvedTests = value; } }
    }
}