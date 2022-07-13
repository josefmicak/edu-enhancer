using DomainModel;

namespace ViewLayer.Models
{
    public class TestTemplateListModel : PageModel
    {
        private List<TestTemplate> testTemplates;

        public List<TestTemplate> TestTemplates { get { return testTemplates; } set { testTemplates = value; } }
    }
}
