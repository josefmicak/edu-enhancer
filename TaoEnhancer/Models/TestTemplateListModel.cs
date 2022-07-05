using DomainModel;

namespace ViewLayer.Models
{
    public class TestTemplateListModel
    {
        private List<TestTemplate> testTemplates = new List<TestTemplate>();

        public List<TestTemplate> TestTemplates { get { return testTemplates; } set { testTemplates = value; } }
    }
}
