namespace ViewLayer.Models
{
    public class TestTemplateListModel : PageModel
    {
        private List<(string, string, string)> tests = new List<(string, string, string)>();
        
        public List<(string, string, string)> Tests { get { return tests; } set { tests = value; } }
    }
}